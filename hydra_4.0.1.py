import asyncio, os, re, csv, glob, random, time, httpx, pandas as pd, emoji, argparse
from datetime import datetime
from playwright.async_api import async_playwright
from langdetect import detect
from multiprocessing import Process, Manager, Lock, Value
from threading import Thread
from tqdm import tqdm

# Worker and retry configuration
NUM_WORKERS = 8
MAX_RETRIES = 3
HEAD_TIMEOUT = 5
TWEET_URL = "https://twitter.com/i/web/status/{}"

# Fail phrase detection (multilingual-ready)
fail_phrases = {
    "protected": [
        "only approved followers",
        "protected tweets",
        "solo los seguidores aprobados",
        "nur genehmigte follower",
        "仅限获批准的关注者",
        "فقط المتابعون المعتمدون"
    ],
    "age_restricted": [
        "might not be appropriate for people under 18",
        "puede no ser apropiado para menores",
        "könnte für Personen unter 18 ungeeignet sein",
        "可能不适合18岁以下的人",
        "قد لا يكون مناسبًا لمن هم دون 18 عامًا"
    ],
    "tweet_unavailable": [
        "this tweet is unavailable",
        "ce tweet est indisponible",
        "este tweet no está disponible",
        "dieser tweet ist nicht verfügbar",
        "此推文不可用",
        "هذه التغريدة غير متوفرة"
    ],
    "account_suspended": [
        "account suspended",
        "compte suspendu",
        "cuenta suspendida",
        "konto gesperrt",
        "账户已暂停",
        "تم تعليق الحساب"
    ],
    "sensitive_content_block": [
        "this media may contain sensitive content",
        "ce média peut contenir du contenu sensible",
        "este contenido puede ser sensible",
        "dieser inhalt kann sensibel sein",
        "此媒体可能包含敏感内容",
        "قد يحتوي هذا الوسائط على محتوى حساس"
    ],
    "deleted": [
        "this page doesn’t exist",
        "try searching for something else",
        "cette page n'existe pas",
        "cette page est introuvable",
        "esta página no existe",
        "diese seite existiert nicht",
        "此页面不存在",
        "هذه الصفحة غير موجودة"
    ]
}

# Junk phrase filtering (UI clutter, cookie prompts, etc.)
JUNK_PHRASES = [
    "accept all cookies", "refuse non-essential cookies", "log in", "open app",
    "see new posts", "download the app", "get the app", "age-restricted adult content",
    "this content might not be appropriate", "learn more", "show more about your choices",
    "sign up", "create account", "view profile", "explore more", "join now", "get started",
    "accepter tous les cookies", "refuser les cookies non essentiels", "se connecter",
    "ouvrir l'application", "ver publicaciones nuevas", "descargar la aplicación",
    "registrarse", "crear cuenta", "查看新帖子", "下载应用", "注册", "创建账户",
    "انضم الآن", "ابدأ", "عرض الملف الشخصي"
]

# Utility: extract date from filename
def extract_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", filename)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")

# Utility: load tweet IDs from input file
def load_valid_tweet_ids(filename):
    ids = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "FILE INFORMATION": break
            if line.isdigit(): ids.append(line)
    return list(dict.fromkeys(ids))

# Utility: collect already scraped tweet IDs
def get_already_scraped_ids():
    scraped = set()
    for file in glob.glob("scraped_worker_*.csv"):
        try:
            df = pd.read_csv(file)
            scraped.update(df["tweet_id"].astype(str).str.strip().str.replace("'", ""))
        except: pass
    return scraped

# Utility: check internet connectivity
def is_online():
    try:
        httpx.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

async def hydrate_tweet(page, tweet_id):
    url = TWEET_URL.format(tweet_id)

    await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media"] else route.continue_())
    await page.route("**/card*", lambda route: route.abort())

    try:
        await page.goto(url, timeout=30000, wait_until="networkidle")
        await page.wait_for_load_state("domcontentloaded")
        for i in range(0, 6000, 500):
            await page.evaluate(f"window.scrollTo(0, {i})")
            await page.wait_for_timeout(600)
        await page.wait_for_timeout(3000)
    except Exception:
        return {
            "text": "",
            "hashtags": "",
            "language": "",
            "media": "no",
            "has_link": "no",
            "fail_reason": "navigation_error",
            "hydration_score": 0,
            "views": "N/A",
            "retweets": "N/A",
            "likes": "N/A",
            "replies": "N/A",
            "quote_count": "N/A"
        }

    fail_reason = ""
    text = ""
    hydration_score = 0

    try:
        html = await page.content()
        full_text = await page.inner_text("body")
        span_texts = await page.locator("span").all_inner_texts()
        combined_text = " ".join([html, full_text] + span_texts).lower()
    except:
        combined_text = ""

    try:
        tweet_element = await page.query_selector('[data-testid="tweetText"]')
        if tweet_element:
            text = await tweet_element.inner_text()
            hydration_score += 3
    except:
        pass

    if not text.strip():
        try:
            og_title = await page.get_attribute('meta[property="og:title"]', "content")
            twitter_title = await page.get_attribute('meta[name="twitter:title"]', "content")
            page_title = await page.title()
            oembed_title = await page.get_attribute('link[rel="alternate"]', "title")
            raw = og_title or twitter_title or oembed_title or page_title or ""
            match = re.search(r'[“"](.+?)[”"]', raw)
            text = match.group(1) if match else raw.strip()
            if "on X:" in text:
                text = text.split("on X:")[1].strip()
            if text:
                hydration_score += 2
        except:
            pass

    if not text.strip():
        try:
            filtered_spans = [s for s in span_texts if s.strip().lower() not in JUNK_PHRASES]
            combined = " ".join(filtered_spans).strip()
            if combined and len(re.findall(r"\w", combined)) >= 15:
                text = combined
                hydration_score += 1
        except:
            pass

    hashtags = ", ".join(re.findall(r"#\w+", text))

    try:
        tweet_container = await page.query_selector("article")
        has_image = await tweet_container.query_selector('[data-testid="tweetPhoto"]') is not None
        has_video = await tweet_container.query_selector('[data-testid="videoPlayer"]') is not None
        has_gif = await tweet_container.query_selector('[data-testid="animatedGif"]') is not None
        media = "yes" if has_image or has_video or has_gif else "no"
    except:
        media = "no"

    has_link = "yes" if "http://" in text or "https://" in text or "t.co/" in text else "no"

    if "#" in text: hydration_score += 1
    if "http" in text or "t.co/" in text: hydration_score += 1
    if "@" in text: hydration_score += 1
    if len(re.findall(r"\w", text)) >= 15: hydration_score += 1

    if hydration_score <= 2:
        for status, phrases in fail_phrases.items():
            for phrase in phrases:
                if phrase.lower() in combined_text:
                    fail_reason = status
                    break
            if fail_reason:
                break

    if fail_reason:
        language = ""
    else:
        try:
            language = detect(text) if len(text) >= 10 else ""
        except:
            language = ""

    # Engagement metrics with fallback
    async def safe_inner_text(selector):
        try:
            el = await page.query_selector(selector)
            return await el.inner_text() if el else "N/A"
        except:
            return "N/A"

    views = await safe_inner_text('[data-testid="viewCount"]')
    retweets = await safe_inner_text('[data-testid="retweet"]')
    likes = await safe_inner_text('[data-testid="like"]')
    replies = await safe_inner_text('[data-testid="reply"]')
    quote_count = await safe_inner_text('[data-testid="quote"]')

    return {
        "text": text,
        "hashtags": hashtags,
        "language": language,
        "media": media,
        "has_link": has_link,
        "fail_reason": fail_reason,
        "hydration_score": hydration_score,
        "views": views,
        "retweets": retweets,
        "likes": likes,
        "replies": replies,
        "quote_count": quote_count
    }

async def worker_main(worker_id, queue, scraped_ids, lock, head_map, progress_counter, throttled_mode):
    output_file = f"scraped_worker_{worker_id}.csv"
    failed = []
    failure_streak = 0
    MAX_FAILURE_STREAK = 20

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--dns-prefetch-disable", "--no-sandbox"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
            viewport={"width": 375, "height": 667},
            device_scale_factor=2,
            is_mobile=True
        )
        page = await context.new_page()

        await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media"] else route.continue_())

        with open(output_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tweet_id", "text", "hashtags", "language", "media", "has_link",
                "views", "retweets", "likes", "replies", "quote_count",
                "URL", "fail_reason", "hydration_score", "summary"
            ])
            writer.writeheader()

            while not queue.empty():
                tweet_id = queue.get()
                if head_map.get(tweet_id) == 404 or tweet_id in scraped_ids:
                    failed.append(f"{tweet_id} — skipped")
                    with progress_counter.get_lock():
                        progress_counter.value += 1
                    continue

                while not is_online():
                    time.sleep(30)

                if throttled_mode:
                    time.sleep(random.uniform(2, 5))

                for attempt in range(MAX_RETRIES):
                    try:
                        result = await hydrate_tweet(page, tweet_id)
                        if result and isinstance(result, dict) and "hydration_score" in result:
                            try:
                                writer.writerow({
                                    "tweet_id": f"'{tweet_id}",
                                    "text": result.get("text", ""),
                                    "hashtags": result.get("hashtags", ""),
                                    "language": result.get("language", ""),
                                    "media": result.get("media", ""),
                                    "has_link": result.get("has_link", ""),
                                    "views": result.get("views", "N/A"),
                                    "retweets": result.get("retweets", "N/A"),
                                    "likes": result.get("likes", "N/A"),
                                    "replies": result.get("replies", "N/A"),
                                    "quote_count": result.get("quote_count", "N/A"),
                                    "URL": f'=HYPERLINK("https://twitter.com/i/web/status/{tweet_id}", "View Tweet")',
                                    "fail_reason": result.get("fail_reason", ""),
                                    "hydration_score": result.get("hydration_score", 0),
                                    "summary": ""
                                })
                                f.flush()
                                break
                            except Exception as e:
                                failed.append(f"{tweet_id} — write error: {str(e)}")
                        else:
                            failed.append(f"{tweet_id} — empty result")
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            failed.append(f"{tweet_id} — {str(e)}")

                if result.get("fail_reason") == "navigation_error":
                    failure_streak += 1
                else:
                    failure_streak = 0

                if failure_streak >= MAX_FAILURE_STREAK:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
                    flag_path = os.path.join(os.getcwd(), f"Hydra was throttled to death at {timestamp}.flag")
                    with open(flag_path, "w") as f_flag:
                        f_flag.write("Throttling threshold exceeded. Worker shut down.\n")
                    print(f"[Worker {worker_id}] Throttle detected — flag file created: {flag_path}")
                    break

                with progress_counter.get_lock():
                    progress_counter.value += 1

        await browser.close()

    if failed:
        with lock:
            with open(f"failed_worker_{worker_id}.txt", "w") as f:
                for tid in failed:
                    f.write(tid + "\n")

def run_worker(worker_id, queue, scraped_ids, lock, head_map, progress_counter, throttled_mode):
    asyncio.run(worker_main(
        worker_id,
        queue,
        scraped_ids,
        lock,
        head_map,
        progress_counter,
        throttled_mode
    ))


def merge_csvs(output_file):
    all_files = glob.glob("scraped_worker_*.csv")
    combined = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    combined.drop_duplicates(subset="tweet_id", inplace=True)

    # Inline summary stats
    lang_counts = combined["language"].value_counts().to_dict()
    fail_counts = combined["fail_reason"].value_counts().to_dict()

    summary_lines = [
        f"Languages: {lang_counts}",
        f"Failures: {fail_counts}",
    ]
    summary_text = " | ".join(summary_lines)

    # Merge-level decluttering: blank low-score or failed tweets
    combined["hydration_score"] = pd.to_numeric(combined["hydration_score"], errors="coerce").fillna(0).astype(int)
    combined["fail_reason"] = combined["fail_reason"].fillna("").astype(str)
    combined.loc[
        (combined["fail_reason"].str.strip() != "") | (combined["hydration_score"] <= 2),
        "text"
    ] = ""

    if not combined.empty:
        combined["summary"] = combined["summary"].astype("object")
        combined.at[0, "summary"] = summary_text

    combined.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nMerged CSV saved as: {output_file}")


def collect_failed_ids(output_file):
    failed_files = glob.glob("failed_worker_*.txt")
    all_failed = set()
    for file in failed_files:
        with open(file) as f:
            for line in f:
                tid = line.strip().split(" — ")[0]
                if tid: all_failed.add(tid)
    if all_failed:
        with open(output_file, "w") as f:
            for tid in sorted(all_failed):
                f.write(tid + "\n")
        print(f"Recovery file created: {output_file} ({len(all_failed)} failed)")

def cleanup_worker_files():
    for pattern in ["scraped_worker_*.csv", "failed_worker_*.txt"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass

def annotate_head_status(tweet_ids):
    async def run():
        head_map = {}
        semaphore = asyncio.Semaphore(100)

        async def limited_check(tid, client):
            async with semaphore:
                try:
                    r = await client.head(TWEET_URL.format(tid), timeout=HEAD_TIMEOUT)
                    return tid, r.status_code
                except:
                    return tid, "timeout"

        try:
            async with httpx.AsyncClient() as client:
                tasks = [limited_check(tid, client) for tid in tweet_ids]
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="HEAD checks"):
                    tid, status = await coro
                    head_map[tid] = status
        except KeyboardInterrupt:
            return head_map

        return head_map

    return asyncio.run(run())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--throttle", action="store_true", help="Enable manual throttle mode")
    args = parser.parse_args()
    throttled_mode = args.throttle

    for tweet_file in sorted(glob.glob("tweet_ids*.txt")):
        date_tag = extract_date_from_filename(tweet_file)
        merged_file = f"scraped_combined_{date_tag}.csv"
        head_map_file = f"head_map_cache_{date_tag}.txt"

        if os.path.exists(merged_file):
            print(f"\nSkipping {tweet_file} — merged file already exists.")
            continue

        print(f"\nStarting hydration for {tweet_file} ({date_tag})")

        all_ids = load_valid_tweet_ids(tweet_file)
        scraped_ids = get_already_scraped_ids()

        # Resume-aware HEAD check using plain text
        head_map = {}
        if os.path.exists(head_map_file):
            print("✔️ Skipping HEAD check — using cached head_map.")
            with open(head_map_file, "r", encoding="utf-8") as f:
                for line in f:
                    tid, status = line.strip().split(",", 1)
                    head_map[tid] = int(status) if status.isdigit() else status
        else:
            print("🆕 No head_map cache found — running HEAD check.")
            head_map = annotate_head_status(all_ids)
            with open(head_map_file, "w", encoding="utf-8") as f:
                for tid, status in head_map.items():
                    f.write(f"{tid},{status}\n")

        # HEAD check summary
        status_summary = {
            "200": 0, "404": 0, "403": 0, "timeout": 0, "other": 0
        }
        for status in head_map.values():
            if status == 200:
                status_summary["200"] += 1
            elif status == 404:
                status_summary["404"] += 1
            elif status == 403:
                status_summary["403"] += 1
            elif status == "timeout":
                status_summary["timeout"] += 1
            else:
                status_summary["other"] += 1

        print("\nHEAD check estimates:")
        print(f"- Total tweets: {len(all_ids)}")
        print(f"- Likely active (200 OK): {status_summary['200']}")
        print(f"- Deleted (404): {status_summary['404']}")
        print(f"- Protected or blocked (403): {status_summary['403']}")
        print(f"- Timed out: {status_summary['timeout']}")
        print(f"- Other responses: {status_summary['other']}")

        proceed = input("\nProceed with hydration? [Y/n]: ").strip().lower()
        if proceed not in ["y", "yes", ""]:
            print("Hydration aborted by user.")
            return

        # Worker setup
        manager = Manager()
        queue = manager.Queue()
        lock = Lock()
        progress_counter = Value("i", 0)

        for tid in all_ids:
            queue.put(tid)

        total = queue.qsize()
        progress_bar = tqdm(total=total, desc="Total Progress", position=0)

        def update_progress():
            while progress_counter.value < total:
                with progress_counter.get_lock():
                    progress_bar.n = progress_counter.value
                progress_bar.refresh()
                time.sleep(0.5)
            progress_bar.n = total
            progress_bar.refresh()
            progress_bar.close()

        monitor = Thread(target=update_progress)
        monitor.start()

        processes = []
        for i in range(NUM_WORKERS):
            p = Process(target=run_worker, args=(i + 1, queue, scraped_ids, lock, head_map, progress_counter, throttled_mode))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        monitor.join()

        merge_csvs(merged_file)
        collect_failed_ids(f"failed_ids_{date_tag}.txt")
        cleanup_worker_files()

        if os.path.exists(head_map_file):
            os.remove(head_map_file)
            print(f" Deleted cache file: {head_map_file}")

if __name__ == "__main__":
    main()

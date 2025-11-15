import os
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import ahocorasick
import re
import unicodedata


global_matchers = None

fragile_acronyms = {"msf", "acf", "nrc", "crs", "pui", "hi", "mdm"}

def split_glued_acronyms(text, acronyms=fragile_acronyms):
    for acro in acronyms:
        text = re.sub(rf"({acro})(?=[a-z])", r"\1 ", str(text), flags=re.IGNORECASE)
    return text

def normalize_text(text):
    text = str(text) if not isinstance(text, str) else text
    text = split_glued_acronyms(text)
    text = re.sub(r"[#@]", "", text.lower())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFKC", text)

def detect_languages(text):
    try:
        langs = detect_langs(text)
        return [lang.lang for lang in langs if lang.prob > 0.2]
    except:
        return ["unknown"]

def assign_language(langs):
    return langs[0] if langs else "unknown"

def is_multi_language(langs):
    return len(langs) > 1

def detect_language(text):
    try:
        text = str(text).strip()
        return detect(text)
    except (LangDetectException, Exception):
        return "unknown"

category_keywords = {
    "red_cross": [
        "red cross", "red crosses", "international committee of the red cross", "icrc", "ifrc",
        "croix-rouge", "croix-rouges", "comité international de la croix-rouge", "cicr",
        "cruz roja", "cruces rojas", "comité internacional de la cruz roja",
        "красный крест", "красные кресты", "международный комитет красного креста",
        "الصليب الأحمر", "اللجنة الدولية للصليب الأحمر",
        "红十字会", "红十字组织", "红十字国际委员会",
        "american red cross", "british red cross", "ukrainian red cross", "german red cross",
        "canadian red cross", "australian red cross", "japanese red cross", "swiss red cross",
        "italian red cross", "french red cross", "norwegian red cross", "swedish red cross",
        "red crescent", "red crescents", "turkish red crescent", "egyptian red crescent",
        "pakistan red crescent", "iranian red crescent", "syrian red crescent",
        "palestinian red crescent", "الهلال الأحمر", "جمعية الهلال الأحمر",
        "الهلال الأحمر التركي", "الهلال الأحمر الفلسطيني",
        "magen david adom", "mda", "מגן דוד אדום"
    ],

    "msf_solidarist": [
        "médecins sans frontières", "msf", "doctors without borders",
        "première urgence internationale", "handicap international",
        "action contre la faim", "acf", "action against hunger",
        "norwegian refugee council", "nrc",
        "doctors of the world", "médecins du monde", "mdm",
        "terre des hommes", "refugees international",
        "human rights watch", "hrw",
        "amnesty international",
        "international crisis group", "crisis group",
        "reporters without borders", "rsf",
        "geneva call",
        "danish refugee council",
        "international medical corps", "save the children",
        "vostok sos", "right to protection",
        "médicos sin fronteras", "acción contra el hambre", "refugiados internacionales",
        "врачи без границ", "врачи мира", "норвежский совет по делам беженцев",
        "правозащитники", "амнисти интернешнл", "международная кризисная группа",
        "أطباء بلا حدود", "أطباء العالم", "العمل ضد الجوع", "المجلس النرويجي للاجئين",
        "منظمة العفو الدولية", "هيومن رايتس ووتش", "مجموعة الأزمات الدولية",
        "无国界医生", "世界医生组织", "对抗饥饿", "挪威难民委员会"
    ],

    "faith_based": [
        "caritas", "caritas ukraine", "caritas-spes", "crs", "catholic relief services",
        "samaritan's purse", "world vision", "adra", "transform ukraine",
        "orthodox christian mission center", "iocc", "international orthodox christian charities",
        "services catholiques de secours", "vision mondiale",
        "servicios católicos de ayuda", "visión mundial",
        "каритас", "каритас украина", "православный христианский миссионерский центр",
        "كاريتاس", "كاريتاس أوكرانيا", "خدمات الإغاثة الكاثوليكية", "رؤية العالم",
        "明爱", "乌克兰明爱", "天主教救援服务", "世界宣明会"
    ],

    "un_agencies": [
        "U.N.", "United Nations", "@UN",
        "UN-led", "UN-backed", "UN-supported", "UN-sponsored", "UN-run",
        "unhcr", "ocha", "iom", "undp", "wfp", "unicef",
        "مفوضية الأمم المتحدة لشؤون اللاجئين", "برنامج الأمم المتحدة الإنمائي",
        "联合国", "联合国开发计划署", "联合国儿童基金会"
    ],
    "usaid": [
        "usaid", "us aid agency",
        "us development agency", "us international aid",
        "chemonics", "dt global", "dexis", "creative associates international"
    ],
    "rossotrudnichestvo": [
        "rossotrudnichestvo", "rossotrudnichestva", "rosstrudnichestvo", "rossotrudnichestva agency",
        "russian aid agency", "russian humanitarian agency", "россотрудничества", "россотруднечество",
        "russian red cross", "russian aid convoy", "russian humanitarian mission"
    ],
    "uncategorized": [
        "razom for ukraine", "nova ukraine"
    ]
}

general_org_terms = [
    "ngo", "ngos",
    "non-governmental organization", "non-governmental organizations",
    "relief agency", "relief agencies",
    "aid group", "aid groups",
    "charity", "charities",
    "volunteer organization", "volunteer organizations",
    "humanitarian organization", "humanitarian organizations",
    "relief organization", "relief organizations",
    "aid agency", "aid agencies"
]

compiled_category_patterns = {
    category: [re.compile(rf"\b{re.escape(keyword.lower())}\b") for keyword in keywords]
    for category, keywords in category_keywords.items()
}

def build_aho_matchers(category_keywords):
    matchers = {}
    for category, keywords in category_keywords.items():
        automaton = ahocorasick.Automaton()
        for keyword in keywords:
            automaton.add_word(keyword, keyword)
        automaton.make_automaton()
        matchers[category] = automaton
    return matchers

aho_matchers = build_aho_matchers(category_keywords)


def categorize_tweet(text):
    try:
        text = str(text).strip()
    except Exception:
        return []

    categories = set()
    normalized = normalize_text(text)
    raw_text = str(text)

    # Strict UN/WHO matching (capitalization-sensitive)
    UN_MATCH = re.search(r"\b(UN|UN-(led|backed|supported|sponsored|run)|U\.N\.|United Nations|@UN|WHO|World Health Organization|@WHO)\b", raw_text)
    if UN_MATCH:
        categories.add("un_agencies")

    # Strict ACTED matching → msf_solidarist
    if re.search(r"\b(ACTED|@ACTED)\b", raw_text):
        categories.add("msf_solidarist")

    # Precompiled keyword matching (normalized)
    for category, patterns in compiled_category_patterns.items():
        for pattern in patterns:
            if pattern.search(normalized):
                categories.add(category)
                break

    return categories

def is_general_org_mention(text):
    norm_text = normalize_text(text)
    return any(term.lower() in norm_text for term in general_org_terms)

def init_worker(matchers):
    global global_matchers
    global_matchers = matchers

def process_chunk(chunk, filename):
    chunk["categories"] = chunk["text"].apply(categorize_tweet)
    chunk["multi_category_match"] = chunk["categories"].apply(lambda cats: len(cats) > 1)
    chunk["language"] = chunk["text"].apply(detect_language)
    return chunk

def log_processed_file(summary_path, full_path, total, matched_total, duplicate_count, multi_category_count):
    summary_row = {
        "file": full_path,
        "total": total,
        "matched": matched_total,
        "duplicates": duplicate_count,
        "multi_category": multi_category_count
    }

    if not os.path.exists(summary_path):
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame([summary_row]).to_csv(summary_path, mode="a", index=False, header=False, encoding="utf-8")

class TweetTally:
    def __init__(self):
        self.total = 0
        self.matched = 0
        self.duplicates = 0
        self.multi_category = 0

    def update(self, batch_total, batch_matched, batch_duplicates, batch_multi):
        self.total += batch_total
        self.matched += batch_matched
        self.duplicates += batch_duplicates
        self.multi_category += batch_multi

    def summary(self):
        rate = self.matched / self.total if self.total else 0
        return (
            f"Total: {self.total} | Matched: {self.matched} | Duplicates: {self.duplicates} "
            f"| Multi-category: {self.multi_category} | Match rate: {rate:.2%}"
        )

ROOT_FOLDER = "tweet_data"
LOG_FOLDER = os.path.join(ROOT_FOLDER, "logs")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "filtered_output")
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_path = os.path.join(OUTPUT_FOLDER, "filter_summary.csv")

def load_processed_files(path):
    if os.path.exists(path):
        try:
            return set(pd.read_csv(path, encoding="utf-8")["file"])
        except Exception:
            return set()
    return set()

output_columns = [
    "userid", "username", "acctdesc", "location", "following", "followers", "totaltweets", "usercreatedts",
    "tweetid", "tweetcreatedts", "retweetcount", "text", "hashtags", "language", "coordinates",
    "favorite_count", "extractedts",
    "source_file", "language", "is_duplicate", "multi_category_match",
    "timezone_assumed"
]

for category in category_keywords.keys():
    for lang_code in ["en", "ru", "other"]:
        path = os.path.join(OUTPUT_FOLDER, f"{category}_{lang_code}.csv")
        if not os.path.exists(path):
            pd.DataFrame(columns=output_columns).to_csv(path, index=False, encoding="utf-8")

tally = TweetTally()
processed_files = load_processed_files(summary_path)

print("Scylla begins her scan...\n")
for dirpath, _, filenames in os.walk(ROOT_FOLDER):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)

        if (
            not filename.endswith(".csv")
            or "_cleaned" in filename
            or "logs" in dirpath
            or "filtered_output" in dirpath
        ):
            continue

        if full_path in processed_files:
            user_input = input(f"Already processed: {filename}. Process again? (yes/no): ").strip().lower()
            if user_input not in ["yes", "y"]:
                print(f"Skipping {filename}")
                continue

        print(f"Processing: {filename}")
        try:
            chunk_iter = pd.read_csv(full_path, dtype=str, encoding="utf-8", chunksize=10000, low_memory=False)

            possible_text_columns = ["text", "tweet", "tweet_text", "content", "tweet_bodz"]
            first_chunk = next(chunk_iter)
            text_column = next((col for col in first_chunk.columns if col.strip().lower() in [name.lower() for name in possible_text_columns]), None)
            if not text_column:
                raise ValueError(f"No recognizable tweet text column found in {filename}")
            print(f"Using column '{text_column}'")

            chunk_iter = pd.read_csv(full_path, dtype=str, encoding="utf-8", chunksize=10000, low_memory=False)

            categorized_ids = set()
            total = 0
            multi_category_count = 0

            for chunk in chunk_iter:
                chunk.rename(columns={text_column: "text"}, inplace=True)
                chunk = chunk.fillna("")
                chunk["is_duplicate"] = False
                
                # Normalize original hydrated language column
                chunk["language"] = chunk["language"].astype(str).str.lower()

                # Fallback detection (stored separately)
                chunk["detected_languages"] = chunk["text"].apply(detect_languages)
                chunk["fallback_language"] = chunk["detected_languages"].apply(assign_language)
                chunk["multi_language"] = chunk["detected_languages"].apply(is_multi_language)

                # Use fallback only if original language is missing or invalid
                valid_langs = ["en", "ru"]
                chunk["language"] = chunk["language"].apply(
                    lambda x: x if x in valid_langs else "other"
)

                if "tweetcreatedts" in chunk.columns:
                    chunk["tweetcreatedts"] = pd.to_datetime(chunk["tweetcreatedts"], errors="coerce", utc=True)
                    chunk["timezone_assumed"] = "UTC"

                chunk["source_file"] = filename
                chunk["categories"] = chunk["text"].apply(categorize_tweet)
                chunk["multi_category_match"] = chunk["categories"].apply(lambda cats: len(cats) > 1)

                for category in category_keywords:
                    matched = chunk[chunk["categories"].apply(lambda cats: category in cats)]
                    if not matched.empty:
                        for lang_code in ["en", "ru"]:
                            lang_df = matched[matched["language"] == lang_code]
                            if not lang_df.empty:
                                path = os.path.join(OUTPUT_FOLDER, f"{category}_{lang_code}.csv")
                                lang_df.reindex(columns=output_columns).to_csv(path, mode="a", index=False, header=False, encoding="utf-8")

                        other_df = matched[~matched["language"].isin(["en", "ru"])]
                        if not other_df.empty:
                            path = os.path.join(OUTPUT_FOLDER, f"{category}_other.csv")
                            other_df.reindex(columns=output_columns).to_csv(path, mode="a", index=False, header=False, encoding="utf-8")

                        categorized_ids.update(matched.index)

                uncategorized = chunk[
                    (~chunk.index.isin(categorized_ids)) &
                    (chunk["text"].apply(is_general_org_mention))
                ]
                if not uncategorized.empty:
                    for lang_code in ["en", "ru"]:
                        lang_df = uncategorized[uncategorized["language"] == lang_code]
                        if not lang_df.empty:
                            path = os.path.join(OUTPUT_FOLDER, f"uncategorized_{lang_code}.csv")
                            lang_df.reindex(columns=output_columns).to_csv(path, mode="a", index=False, header=False, encoding="utf-8")

                    other_df = uncategorized[~uncategorized["language"].isin(["en", "ru"])]
                    if not other_df.empty:
                        path = os.path.join(OUTPUT_FOLDER, f"uncategorized_other.csv")
                        other_df.reindex(columns=output_columns).to_csv(path, mode="a", index=False, header=False, encoding="utf-8")

                    categorized_ids.update(uncategorized.index)

                total += len(chunk)
                multi_category_count += chunk["multi_category_match"].sum()

            matched_total = len(categorized_ids)
            duplicate_count = 0  # placeholder; dedup happens later
            tally.update(total, matched_total, duplicate_count, multi_category_count)

            log_processed_file(summary_path, full_path, total, matched_total, duplicate_count, multi_category_count)
            print(f" {filename} | Total: {total} | Matched: {matched_total} | Duplicates: {duplicate_count} | Multi-category: {multi_category_count}")
            print(tally.summary() + "\n")

        except Exception as e:
            print(f"Error in {full_path}: {e}\n")
print(f"\n Filtering complete. Summary saved to: {summary_path}")
print(tally.summary())
print("Scylla has finished her scan.")

def deduplicate_outputs(folder):
    print("\nStarting post-run deduplication...")
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8")
            before = len(df)
            if "tweetid" in df.columns:
                df["is_duplicate"] = df.duplicated(subset="tweetid", keep="first")
                df = df[~df["is_duplicate"]]
            else:
                df["is_duplicate"] = False
            df.to_csv(path, index=False, encoding="utf-8")
            print(f"Deduplicated {fname}: {before} → {len(df)}")
        except Exception as e:
            print(f"Error deduplicating {fname}: {e}")


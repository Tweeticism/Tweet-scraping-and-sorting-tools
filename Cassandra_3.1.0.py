import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import pipeline
import torch
import hdbscan   # <-- added import

# --- Config -------------------------------------------------------------------
FILE_PATH = r"C:\Users\UNIGE\Desktop\Notes\dissertation_communication_related\Misinformation Research\Data Set\ForTopicModeling\unicef_tweets.csv"
TEXT_COL = "text"
TIME_COL = "tweetcreatedts"
LANG_COL_PRIMARY = "language"
LANG_COL_FALLBACK = "fil_language"
KEEP_LANG = {"en"}
DROP_RETWEETS = True
DROP_DUPLICATES_FLAG = "is_duplica"
MIN_DOC_LEN = 5
MIN_DF = 2
MAX_DF = 0.95
OUT_DIR = Path("analysis_outputs")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# --- Helpers ------------------------------------------------------------------
def clean_tweet(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def is_retweet(text: str, retweetco):
    rt_flag = isinstance(text, str) and text.startswith("RT ")
    count_flag = (retweetco is not None) and (pd.notna(retweetco)) and (retweetco > 0)
    return rt_flag or count_flag

# --- Load ---------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(FILE_PATH)

# Language filtering
if KEEP_LANG and (LANG_COL_PRIMARY in df.columns or LANG_COL_FALLBACK in df.columns):
    lang_series = None
    if LANG_COL_PRIMARY in df.columns:
        lang_series = df[LANG_COL_PRIMARY].astype(str).str.lower()
    elif LANG_COL_FALLBACK in df.columns:
        lang_series = df[LANG_COL_FALLBACK].astype(str).str.lower()
    df = df[lang_series.isin({l.lower() for l in KEEP_LANG})].copy()

# Timestamp parsing
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.dropna(subset=[TEXT_COL, TIME_COL]).copy()

# Drop duplicates
if DROP_DUPLICATES_FLAG and DROP_DUPLICATES_FLAG in df.columns:
    mask = ~(df[DROP_DUPLICATES_FLAG].astype(str).str.lower().isin({"true", "1"}))
    df = df[mask].copy()

# Drop retweets
if DROP_RETWEETS:
    retweetco = df["retweetco"] if "retweetco" in df.columns else None
    df = df[~df.apply(lambda r: is_retweet(r[TEXT_COL], retweetco.iloc[r.name] if retweetco is not None else None), axis=1)].copy()

# Clean and filter
df["clean_text"] = df[TEXT_COL].apply(clean_tweet)
df = df[df["clean_text"].str.len() >= MIN_DOC_LEN].drop_duplicates(subset=["clean_text"]).copy()

texts = df["clean_text"].tolist()
timestamps = df[TIME_COL].tolist()
print(f"Documents ready: {len(texts)}")

# --- Embeddings ---------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
if device == "cuda":
    try:
        embedder.half()
        print("FP16 enabled.")
    except Exception:
        pass

# Batched embeddings
embeddings = embedder.encode(
    texts,
    batch_size=256,              # adjust based on GPU memory
    show_progress_bar=True
)

# --- BERTopic -----------------------------------------------------------------
vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=MIN_DF, max_df=MAX_DF, stop_words="english")

# Explicit HDBSCAN with min_cluster_size=5
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5)

topic_model = BERTopic(
    embedding_model=embedder,
    vectorizer_model=vectorizer_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts, embeddings)
df["topic"] = topics
df["prob"] = probs

# Save core outputs
df.to_csv(OUT_DIR / "tweets_with_topics.csv", index=False)
topic_info = topic_model.get_topic_info()
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False)

# Representative docs
rep_docs = topic_model.get_representative_docs()
rep_rows = [{"topic": tid, "rank": i + 1, "doc": doc} for tid, docs in rep_docs.items() for i, doc in enumerate(docs)]
pd.DataFrame(rep_rows).to_csv(OUT_DIR / "topic_representative_docs.csv", index=False)

# --- Sentiment overlay (batched GPU) ------------------------------------------
print("Running sentiment analysis...")
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=0 if device == "cuda" else -1)

results = sentiment_pipeline(texts, batch_size=64, truncation=True)
df["sentiment"] = [res["label"] for res in results]
df.to_csv(OUT_DIR / "tweets_with_topics_and_sentiment.csv", index=False)

sentiment_by_topic = (
    df.groupby("topic")["sentiment"]
      .value_counts(normalize=True)
      .unstack()
      .fillna(0.0)
      .sort_index()
)
sentiment_by_topic.to_csv(OUT_DIR / "sentiment_by_topic.csv")

# --- Model persistence ---------------------------------------------------------
topic_model.save(OUT_DIR / "bertopic_model")

# --- Visualizations (safe wrappers) -------------------------------------------
try:
    topic_model.visualize_topics().write_html(OUT_DIR / "intertopic_map.html")
    topic_model.visualize_barchart().write_html(OUT_DIR / "barchart.html")
except Exception as e:
    print(f"Visualization issue: {e}")

try:
    topics_over_time = topic_model.topics_over_time(
        docs=texts,
        topics=topics,
        timestamps=timestamps,
        nr_bins=5   # fewer bins for small datasets
    )
    topics_over_time.to_csv(OUT_DIR / "topics_over_time.csv", index=False)
    topic_model.visualize_topics_over_time(topics_over_time).write_html(OUT_DIR / "topics_over_time.html")
except Exception as e:
    print(f"Timeline visualization issue: {e}")

print("✅ Done. Outputs saved in:", OUT_DIR.resolve())

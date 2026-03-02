# selection/standardize_cached.py
from __future__ import annotations
import pandas as pd

OPTIONAL_COLS = [
    "n_reviews_used",
    "avg_rating_used",
    "n_chars",
    "n_sentences_used",
    "n_words_used",
    "method",
    "params",
]

def standardize_cached_corpus(df: pd.DataFrame, dedupe: str = "best_text") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["place_id", "text"] + OPTIONAL_COLS)

    df = df.copy()

    # required columns check
    if "place_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Cached corpus missing place_id/text. cols={list(df.columns)}")

    # types + cleanup
    df["place_id"] = pd.to_numeric(df["place_id"], errors="coerce")
    df = df.dropna(subset=["place_id"])
    df["place_id"] = df["place_id"].astype(int)

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0]

    # ensure optional columns exist
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # ensure n_chars exists
    if df["n_chars"].isna().all():
        df["n_chars"] = df["text"].str.len()

    # dedupe to one row per place_id
    if df["place_id"].duplicated().any():
        if dedupe == "best_text":
            df["_len"] = df["text"].str.len()
            df = df.sort_values(["place_id", "_len"], ascending=[True, False])
            df = df.drop_duplicates("place_id", keep="first").drop(columns=["_len"])
        else:
            df = df.drop_duplicates("place_id", keep="first")

    # stable order
    base = ["place_id", "text"] + OPTIONAL_COLS
    extra = [c for c in df.columns if c not in base]
    return df[base + extra]
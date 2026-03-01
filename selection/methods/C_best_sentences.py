import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import sent_tokenize

def build(df_models: pd.DataFrame, n: int = 15, max_sentences: int = 120, **params) -> pd.DataFrame:
    nltk.download("punkt", quiet=True)

    rows = []
    for place_id, sub in df_models.groupby("place_id"):
        full_text = " ".join(sub["review_text"].astype(str).tolist())

        sents = [s.strip() for s in sent_tokenize(full_text) if len(s.strip()) >= 15]
        if not sents:
            rows.append((place_id, "", 0, sub.shape[0], float(sub["score"].mean())))
            continue

        # cap for speed
        if len(sents) > max_sentences:
            sents = sents[:max_sentences]

        # If too few sentences, TF-IDF with max_df filtering can break → fallback
        if len(sents) < 3:
            chosen = sents[:min(n, len(sents))]
            out_text = " ".join(chosen)
            rows.append((place_id, out_text, len(chosen), sub.shape[0], float(sub["score"].mean())))
            continue

        # Use TF-IDF over sentences; for safety don't use max_df here
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
            min_df=1
        )

        X = vec.fit_transform(sents)
        scores = np.asarray(X.sum(axis=1)).ravel()

        top_idx = np.argsort(-scores)[:min(n, len(sents))]
        top_idx = sorted(top_idx.tolist())  # keep original order
        chosen = [sents[i] for i in top_idx]

        out_text = " ".join(chosen)
        rows.append((place_id, out_text, len(chosen), sub.shape[0], float(sub["score"].mean())))

    out = pd.DataFrame(rows, columns=["place_id", "text", "n_sentences_used", "n_reviews_used", "avg_rating_used"])
    return out
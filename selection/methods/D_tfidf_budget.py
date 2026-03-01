import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import sent_tokenize

def build(df_models: pd.DataFrame, budget: int = 120, max_sentences: int = 250, **params) -> pd.DataFrame:
    nltk.download("punkt", quiet=True)

    rows = []
    for place_id, sub in df_models.groupby("place_id"):
        full_text = " ".join(sub["review_text"].astype(str).tolist()).lower()

        # split into sentences so TF-IDF has multiple "docs"
        sents = [s.strip() for s in sent_tokenize(full_text) if s.strip()]

        if not sents:
            rows.append((place_id, "", 0, sub.shape[0], float(sub["score"].mean())))
            continue

        # cap for speed
        if len(sents) > max_sentences:
            sents = sents[:max_sentences]

        # if too few sentences, fallback to a simpler approach (top frequent words)
        if len(sents) < 2:
            words = [w for w in full_text.split() if w.isalpha()]
            top = pd.Series(words).value_counts().head(budget).index.tolist()
            out_text = " ".join(top)
            rows.append((place_id, out_text, len(top), sub.shape[0], float(sub["score"].mean())))
            continue

        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
            min_df=1
        )
        X = vec.fit_transform(sents)
        vocab = np.array(vec.get_feature_names_out())
        scores = np.asarray(X.sum(axis=0)).ravel()

        top_idx = np.argsort(-scores)[:min(budget, len(scores))]
        chosen = vocab[top_idx].tolist()

        out_text = " ".join(chosen)
        rows.append((place_id, out_text, len(chosen), sub.shape[0], float(sub["score"].mean())))

    out = pd.DataFrame(rows, columns=["place_id", "text", "n_words_used", "n_reviews_used", "avg_rating_used"])
    return out
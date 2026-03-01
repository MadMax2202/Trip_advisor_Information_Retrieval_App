import time
from pathlib import Path

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def build(
    df_models: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    clusters: int = 8,
    max_sentences: int = 120,
    min_sentence_chars: int = 15,
    cache_root: str = "data/processed_data/selection_cache/dense_extractive_place_cache",
    sleep_s: float = 0.0,
    **params
) -> pd.DataFrame:
    """
    Dense extractive summarization with PER-PLACE caching.
    Cache files:
      {cache_root}/{model_name}/{clusters}/{place_id}.txt
    """
    nltk.download("punkt", quiet=True)
    model = SentenceTransformer(model_name)

    cache_dir = Path(cache_root) / model_name.replace("/", "_").replace(":", "_") / f"clusters_{clusters}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for place_id, sub in df_models.groupby("place_id"):
        place_id = int(place_id)
        cache_file = cache_dir / f"{place_id}.txt"

        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8").strip()
            n_sent_used = None
        else:
            full_text = " ".join(sub["review_text"].astype(str).tolist())
            sents = [s.strip() for s in sent_tokenize(full_text) if len(s.strip()) >= min_sentence_chars]

            if not sents:
                text = ""
                n_sent_used = 0
            else:
                if len(sents) > max_sentences:
                    sents = sents[:max_sentences]

                emb = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)

                k = min(clusters, len(sents))
                if k <= 1:
                    chosen = [sents[0]]
                else:
                    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                    labels = km.fit_predict(emb)
                    centers = km.cluster_centers_

                    chosen_idx = []
                    for c in range(k):
                        idx = np.where(labels == c)[0]
                        if len(idx) == 0:
                            continue
                        sims = emb[idx] @ centers[c]
                        chosen_idx.append(idx[int(np.argmax(sims))])

                    chosen_idx = sorted(set(chosen_idx))
                    chosen = [sents[i] for i in chosen_idx]

                text = " ".join(chosen)
                n_sent_used = len(chosen)

            cache_file.write_text(text, encoding="utf-8")
            if sleep_s > 0:
                time.sleep(sleep_s)

        rows.append((place_id, text, n_sent_used, sub.shape[0], float(sub["score"].mean())))

    out = pd.DataFrame(rows, columns=["place_id", "text", "n_sentences_used", "n_reviews_used", "avg_rating_used"])
    return out
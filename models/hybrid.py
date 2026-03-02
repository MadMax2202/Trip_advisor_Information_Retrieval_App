import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.bm25 import BM25Retriever
from models.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, alpha=0.5, dense_model="all-MiniLM-L6-v2"):
        self.alpha = alpha
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=dense_model)
        self.place_ids = None

    def fit(self, df_test):
        df = df_test.copy()
        df["place_id"] = df["place_id"].astype(int)
        df["text"] = df["text"].fillna("").astype(str)

        self.place_ids = df["place_id"].tolist()
        self.bm25.fit(df[["place_id", "text"]])
        self.dense.fit(df[["place_id", "text"]])

    def rank(self, query_text, exclude_id=None):
        q_text = str(query_text)

        # BM25 scores (match BM25 preprocessing: lowercase split)
        bm25_scores = self.bm25.bm25.get_scores(q_text.lower().split())

        # Dense scores
        q = self.dense.model.encode([q_text])
        dense_scores = cosine_similarity(q, self.dense.doc_embeddings).flatten()

        # Normalize safely
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)

        combined = self.alpha * bm25_scores + (1 - self.alpha) * dense_scores
        ranked_idx = np.argsort(combined)[::-1]
        ranked = [self.place_ids[i] for i in ranked_idx]

        if exclude_id is not None:
            exclude_id = int(exclude_id)
            ranked = [pid for pid in ranked if pid != exclude_id]
        return ranked
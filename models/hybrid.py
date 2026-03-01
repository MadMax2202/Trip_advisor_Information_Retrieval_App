# Hybrid Model (BM25 + Dense)
import numpy as np
from models.bm25 import BM25Retriever
from models.dense import DenseRetriever


class HybridRetriever:

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.place_ids = None

    def fit(self, df_test):
        self.place_ids = df_test["place_id"].tolist()
        self.bm25.fit(df_test)
        self.dense.fit(df_test)

    def rank(self, query_text):

        # BM25 scores
        bm25_scores = self.bm25.bm25.get_scores(query_text.split())

        # Dense scores
        query_emb = self.dense.model.encode([query_text])
        dense_scores = self.dense.model.similarity(query_emb, self.dense.doc_embeddings)[0]

        # Normalize
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)

        combined = self.alpha * bm25_scores + (1 - self.alpha) * dense_scores

        ranked_indices = np.argsort(combined)[::-1]

        return [self.place_ids[i] for i in ranked_indices]
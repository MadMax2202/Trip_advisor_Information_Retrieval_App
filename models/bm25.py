# models/bm25.py
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.place_ids = None

    def fit(self, df_candidates):
        """
        Fit BM25 on candidate set (one row per place).
        Required columns: ['place_id', 'text']
        """
        df_candidates = df_candidates.copy()
        df_candidates["place_id"] = df_candidates["place_id"].astype(int)
        self.place_ids = df_candidates["place_id"].tolist()

        tokenized = [str(t).lower().split() for t in df_candidates["text"].fillna("")]
        self.bm25 = BM25Okapi(tokenized)

    def rank(self, query_text, exclude_id=None):
        """
        Return ranked place_ids best->worst.
        """
        q = str(query_text).lower().split()
        scores = self.bm25.get_scores(q)
        order = np.argsort(scores)[::-1]
        ranked = [self.place_ids[i] for i in order]

        if exclude_id is not None:
            exclude_id = int(exclude_id)
            ranked = [pid for pid in ranked if pid != exclude_id]
        return ranked
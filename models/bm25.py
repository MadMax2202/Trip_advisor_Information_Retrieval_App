import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.bm25 = None
        self.place_ids = None

    def fit(self, df_test):
        """
        Fit BM25 on the candidate (test) set.
        """
        self.place_ids = df_test["place_id"].tolist()
        tokenized_corpus = [doc.split() for doc in df_test["text"]]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def rank(self, query_text):
        """
        Rank test places given a query text.
        Returns ordered list of place_ids.
        """
        scores = self.bm25.get_scores(query_text.split())
        ranked_indices = np.argsort(scores)[::-1]
        return [self.place_ids[i] for i in ranked_indices]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = None
        self.doc_matrix = None
        self.place_ids = None

    def fit(self, df_test):
        df = df_test.copy()
        df["place_id"] = df["place_id"].astype(int)

        self.place_ids = df["place_id"].tolist()
        texts = df["text"].fillna("").astype(str)

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def rank(self, query_text, exclude_id=None):
        q = self.vectorizer.transform([str(query_text)])
        scores = cosine_similarity(q, self.doc_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1]
        ranked = [self.place_ids[i] for i in ranked_idx]

        if exclude_id is not None:
            exclude_id = int(exclude_id)
            ranked = [pid for pid in ranked if pid != exclude_id]
        return ranked
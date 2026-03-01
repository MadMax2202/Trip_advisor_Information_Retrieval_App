import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:

    def __init__(self):
        self.vectorizer = None
        self.doc_matrix = None
        self.place_ids = None

    def fit(self, df_test):
        self.place_ids = df_test["place_id"].tolist()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000
        )

        self.doc_matrix = self.vectorizer.fit_transform(df_test["text"])

    def rank(self, query_text):

        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        ranked_indices = np.argsort(scores)[::-1]

        return [self.place_ids[i] for i in ranked_indices]
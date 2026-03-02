import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.place_ids = None

    def fit(self, df_test):
        df = df_test.copy()
        df["place_id"] = df["place_id"].astype(int)

        self.place_ids = df["place_id"].tolist()
        texts = df["text"].fillna("").astype(str).tolist()

        self.doc_embeddings = self.model.encode(texts, show_progress_bar=True)

    def rank(self, query_text, exclude_id=None):
        q = self.model.encode([str(query_text)])
        scores = cosine_similarity(q, self.doc_embeddings).flatten()
        ranked_idx = np.argsort(scores)[::-1]
        ranked = [self.place_ids[i] for i in ranked_idx]

        if exclude_id is not None:
            exclude_id = int(exclude_id)
            ranked = [pid for pid in ranked if pid != exclude_id]
        return ranked
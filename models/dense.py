# Dense Embeddings (Sentence-BERT)
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.place_ids = None

    def fit(self, df_test):
        self.place_ids = df_test["place_id"].tolist()
        texts = df_test["text"].tolist()
        self.doc_embeddings = self.model.encode(texts, show_progress_bar=True)

    def rank(self, query_text):
        query_embedding = self.model.encode([query_text])
        scores = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        return [self.place_ids[i] for i in ranked_indices]
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


class DenseRerankRetriever:
    """
    Stage 1: Dense bi-encoder retrieves top_k candidates
    Stage 2: Cross-encoder reranks those top_k

    Fits the same interface:
      fit(df[['place_id','text']])
      rank(query_text, exclude_id=None) -> list[place_id]
    """

    def __init__(
        self,
        bi_encoder_name="all-MiniLM-L6-v2",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=50,
    ):
        self.bi = SentenceTransformer(bi_encoder_name)
        self.ce = CrossEncoder(cross_encoder_name)
        self.top_k = int(top_k)

        self.place_ids = None
        self.texts = None
        self.doc_emb = None

    def fit(self, df_test):
        df = df_test.copy()
        df["place_id"] = df["place_id"].astype(int)

        self.place_ids = df["place_id"].tolist()
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.doc_emb = self.bi.encode(self.texts, show_progress_bar=True)

    def rank(self, query_text, exclude_id=None):
        qtext = str(query_text)

        # Stage 1: dense retrieve
        qemb = self.bi.encode([qtext])
        scores = cosine_similarity(qemb, self.doc_emb).flatten()
        top_idx = np.argsort(scores)[::-1][: self.top_k]

        # Stage 2: cross-encoder rerank
        pairs = [(qtext, self.texts[i]) for i in top_idx]
        ce_scores = self.ce.predict(pairs)  # higher = better
        rerank_order = np.argsort(ce_scores)[::-1]
        ranked = [self.place_ids[top_idx[j]] for j in rerank_order]

        if exclude_id is not None:
            exclude_id = int(exclude_id)
            ranked = [pid for pid in ranked if pid != exclude_id]
        return ranked
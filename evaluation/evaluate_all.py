from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from selection.cache import load_or_build
from selection.registry import iter_methods
from selection.standardize_cached import standardize_cached_corpus

from models.bm25 import BM25Retriever
from models.tfidf import TFIDFRetriever
from models.dense import DenseRetriever
from models.hybrid import HybridRetriever
from models.dense_rerank import DenseRerankRetriever   # ✅ ADD THIS

CACHE_DIR = "data/processed_data/selection_cache"


# -------------------------
# LEVEL 2 CATEGORY FUNCTION
# -------------------------
def get_level2_categories_from_row(row: pd.Series) -> set[str]:
    if row is None or row.empty:
        return set()

    typeR = row.get("typeR")
    categories = set()

    if typeR in ["A", "AP"]:
        for col in ["activiteSubCategorie", "activiteSubType"]:
            val = row.get(col)
            if pd.notna(val):
                categories.update(str(val).split(","))

    elif typeR == "R":
        for col in ["restaurantType", "restaurantTypeCuisine"]:
            val = row.get(col)
            if pd.notna(val):
                categories.update(str(val).split(","))

    elif typeR == "H":
        val = row.get("priceRange")
        if pd.notna(val):
            categories.add(str(val))

    return {cat.strip() for cat in categories if cat and str(cat).strip()}


def build_relevance_sets(df_eval: pd.DataFrame):
    df_eval = df_eval.copy()
    df_eval["id"] = pd.to_numeric(df_eval["id"], errors="coerce").astype("Int64")
    df_eval = df_eval.dropna(subset=["id"])
    df_eval["id"] = df_eval["id"].astype(int)

    type_dict = dict(zip(df_eval["id"], df_eval["typeR"]))

    lvl2_dict = {}
    for _, row in df_eval.iterrows():
        pid = int(row["id"])
        lvl2_dict[pid] = get_level2_categories_from_row(row)

    # L1: same typeR
    relL1_dict = {}
    by_type = df_eval.groupby("typeR")["id"].apply(list).to_dict()
    for pid, t in type_dict.items():
        relL1_dict[pid] = set(int(x) for x in by_type.get(t, [])) - {pid}

    # L2: overlap categories
    inv = {}
    for pid, cats in lvl2_dict.items():
        for c in cats:
            inv.setdefault(c, set()).add(pid)

    relL2_dict = {}
    for pid, cats in lvl2_dict.items():
        if not cats:
            relL2_dict[pid] = set()
            continue
        cand = set()
        for c in cats:
            cand |= inv.get(c, set())
        relL2_dict[pid] = cand - {pid}

    return df_eval, relL1_dict, relL2_dict


# -------------------------
# METRICS
# -------------------------
def ranking_error_first_rel(ranked_ids, rel_set):
    for r, pid in enumerate(ranked_ids):
        if pid in rel_set:
            return r
    return len(ranked_ids)

def precision_at_k(ranked_ids, rel_set, k):
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    return sum(1 for pid in topk if pid in rel_set) / float(k)

def recall_at_k(ranked_ids, rel_set, k):
    if not rel_set:
        return 0.0
    topk = ranked_ids[:k]
    return sum(1 for pid in topk if pid in rel_set) / float(len(rel_set))

def hit_rate_at_k(ranked_ids, rel_set, k):
    return 1.0 if any(pid in rel_set for pid in ranked_ids[:k]) else 0.0

def reciprocal_rank(ranked_ids, rel_set):
    for r, pid in enumerate(ranked_ids, start=1):
        if pid in rel_set:
            return 1.0 / float(r)
    return 0.0

def ndcg_at_k(ranked_ids, gain_map, k):
    def dcg(ids):
        s = 0.0
        for i, pid in enumerate(ids[:k], start=1):
            g = gain_map.get(pid, 0.0)
            if g > 0:
                s += (2.0**g - 1.0) / np.log2(i + 1.0)
        return s

    actual = dcg(ranked_ids)
    ideal_ids = sorted(gain_map.keys(), key=lambda x: gain_map[x], reverse=True)
    ideal = dcg(ideal_ids)
    return 0.0 if ideal == 0 else actual / ideal


# -------------------------
# Cache loader (cache-only)
# -------------------------
def load_corpus_from_cache(method_name: str, params: dict) -> pd.DataFrame:
    def _builder():
        raise RuntimeError(
            f"Cache miss for {method_name} {params}. Re-run scripts/build_corpora.py"
        )
    out, meta, hit = load_or_build(CACHE_DIR, method_name, params, _builder)
    if not hit:
        raise RuntimeError("Unexpected cache miss.")
    return out


def build_model_registry():
    return {
        # A) BM25
        "bm25": lambda: BM25Retriever(),

        # B) TF-IDF
        "tfidf": lambda: TFIDFRetriever(),

        # C) Dense (3 variants)
        "dense_allMiniLM": lambda: DenseRetriever("all-MiniLM-L6-v2"),
        "dense_multiqa": lambda: DenseRetriever("multi-qa-MiniLM-L6-cos-v1"),
        "dense_mpnet": lambda: DenseRetriever("all-mpnet-base-v2"),

        # D) Hybrid
        "hybrid_alpha0.5": lambda: HybridRetriever(alpha=0.5, dense_model="all-MiniLM-L6-v2"),

        # E) Dense + Reranking (Cross-Encoder)
        "dense_rerank_top50": lambda: DenseRerankRetriever(
            bi_encoder_name="all-MiniLM-L6-v2",
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=50,
        ),
        # Optional heavier variant:
        # "dense_rerank_top100": lambda: DenseRerankRetriever(
        #     bi_encoder_name="all-MiniLM-L6-v2",
        #     cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        #     top_k=100,
        # ),
    }


def main():
    # -------------------------
    # CONFIG: put your real top5 method names here
    # -------------------------
    TOP5_METHOD_NAMES = [
        # replace with your real top 5
        "D_tfidf_80",
        "D_tfidf_160",
        "D_tfidf_120",
        "B_medium_k10",
        "B_medium_k20",
    ]

    K_LIST = [1, 5, 10, 20]
    RANDOM_STATE = 42
    TEST_SIZE = 0.5
    MAX_QUERIES = None       # set None for full
    MIN_PLACES = 50

    HEARTBEAT_EVERY_EVAL = 25
    HEARTBEAT_EVERY_TRY = 50

    print("Loading df_eval...")
    df_eval_raw = pd.read_csv("data/Tripadvisor.csv", low_memory=False)
    df_eval, relL1_dict, relL2_dict = build_relevance_sets(df_eval_raw)
    allowed_ids = set(df_eval["id"].tolist())

    # method_name -> params
    method_params = {m: params for (m, _module, params) in iter_methods()}

    missing = [m for m in TOP5_METHOD_NAMES if m not in method_params]
    if missing:
        raise ValueError(
            f"TOP5 method names not found in iter_methods(): {missing}\n"
            f"Valid examples: {list(method_params.keys())[:20]} ..."
        )

    models = build_model_registry()
    results = []

    print("\nRunning models on TOP5 cached corpora...")
    for method_name in TOP5_METHOD_NAMES:
        params = method_params[method_name]

        print("\n===================================")
        print("Selection method:", method_name)

        df_corpus = load_corpus_from_cache(method_name, params)
        df_corpus = standardize_cached_corpus(df_corpus, dedupe="best_text")
        df_corpus = df_corpus[df_corpus["place_id"].isin(allowed_ids)]

        n_places = df_corpus["place_id"].nunique()
        if n_places < MIN_PLACES:
            print(f"Skipping: only {n_places} places.")
            continue

        # enforce 1 row per place_id (important for ranking stability)
        df_corpus = df_corpus.sort_values("place_id").drop_duplicates("place_id", keep="first")

        # split by unique place ids
        place_ids = df_corpus["place_id"].unique()
        train_ids, test_ids = train_test_split(
            place_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        df_train = df_corpus[df_corpus["place_id"].isin(train_ids)]
        df_test  = df_corpus[df_corpus["place_id"].isin(test_ids)]

        candidate_set = set(df_test["place_id"].tolist())

        # queries from train
        id_to_text = dict(zip(df_train["place_id"], df_train["text"]))
        query_ids = list(id_to_text.keys())

        rng = np.random.RandomState(RANDOM_STATE)
        rng.shuffle(query_ids)
        if MAX_QUERIES is not None:
            query_ids = query_ids[:MAX_QUERIES]

        print(f"Using {len(query_ids)} queries (MAX_QUERIES={MAX_QUERIES}). Candidates={len(candidate_set)}")

        for model_name, ctor in models.items():
            print("\n--- Model:", model_name)

            retriever = ctor()
            retriever.fit(df_test[["place_id", "text"]])

            # accumulators
            n_used = 0
            tried = 0

            sum_err_l1 = sum_err_l2 = 0.0
            sum_mrr_l1 = sum_mrr_l2 = 0.0

            sum_p_l1 = {k: 0.0 for k in K_LIST}
            sum_r_l1 = {k: 0.0 for k in K_LIST}
            sum_h_l1 = {k: 0.0 for k in K_LIST}

            sum_p_l2 = {k: 0.0 for k in K_LIST}
            sum_r_l2 = {k: 0.0 for k in K_LIST}
            sum_h_l2 = {k: 0.0 for k in K_LIST}

            sum_ndcg = {k: 0.0 for k in K_LIST}

            for qid in query_ids:
                tried += 1
                if tried % HEARTBEAT_EVERY_TRY == 0:
                    print(f"  Heartbeat: tried {tried}/{len(query_ids)} (evaluated {n_used})")

                ranked_ids = retriever.rank(id_to_text[qid], exclude_id=qid)

                relL1 = relL1_dict.get(qid, set()) & candidate_set
                relL2 = relL2_dict.get(qid, set()) & candidate_set

                if not relL1:
                    continue

                sum_err_l1 += ranking_error_first_rel(ranked_ids, relL1)
                sum_err_l2 += ranking_error_first_rel(ranked_ids, relL2) if relL2 else len(ranked_ids)

                sum_mrr_l1 += reciprocal_rank(ranked_ids, relL1)
                sum_mrr_l2 += reciprocal_rank(ranked_ids, relL2) if relL2 else 0.0

                for k in K_LIST:
                    sum_p_l1[k] += precision_at_k(ranked_ids, relL1, k)
                    sum_r_l1[k] += recall_at_k(ranked_ids, relL1, k)
                    sum_h_l1[k] += hit_rate_at_k(ranked_ids, relL1, k)

                    sum_p_l2[k] += precision_at_k(ranked_ids, relL2, k) if relL2 else 0.0
                    sum_r_l2[k] += recall_at_k(ranked_ids, relL2, k) if relL2 else 0.0
                    sum_h_l2[k] += hit_rate_at_k(ranked_ids, relL2, k) if relL2 else 0.0

                # gains: L1=1, L2=2
                gain_map = {pid: 1.0 for pid in relL1}
                for pid in relL2:
                    gain_map[pid] = 2.0
                for k in K_LIST:
                    sum_ndcg[k] += ndcg_at_k(ranked_ids, gain_map, k)

                n_used += 1
                if n_used % HEARTBEAT_EVERY_EVAL == 0:
                    print(f"  Progress: evaluated {n_used}/{len(query_ids)} queries")

            if n_used == 0:
                print("  Skipping model: no valid queries.")
                continue

            row = {
                "method": method_name,
                "model": model_name,
                "n_queries": n_used,
                "ranking_error_l1": sum_err_l1 / n_used,
                "ranking_error_l2": sum_err_l2 / n_used,
                "mrr_l1": sum_mrr_l1 / n_used,
                "mrr_l2": sum_mrr_l2 / n_used,
            }
            for k in K_LIST:
                row[f"P@{k}_L1"] = sum_p_l1[k] / n_used
                row[f"R@{k}_L1"] = sum_r_l1[k] / n_used
                row[f"Hit@{k}_L1"] = sum_h_l1[k] / n_used

                row[f"P@{k}_L2"] = sum_p_l2[k] / n_used
                row[f"R@{k}_L2"] = sum_r_l2[k] / n_used
                row[f"Hit@{k}_L2"] = sum_h_l2[k] / n_used

                row[f"nDCG@{k}"] = sum_ndcg[k] / n_used

            results.append(row)
            print("  Done.", model_name, "| nDCG@10:", row.get("nDCG@10"), "| mrr_l2:", row.get("mrr_l2"))

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results produced.")
        return

    results_df = results_df.sort_values(
        by=["nDCG@10", "mrr_l2", "ranking_error_l2"],
        ascending=[False, False, True],
    )

    os.makedirs("results", exist_ok=True)
    out_path = "results/models_top5_selection_methods.csv"
    results_df.to_csv(out_path, index=False)

    print("\nSaved results:")
    print(" -", out_path)
    print("\nTop rows:")
    print(results_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
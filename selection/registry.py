from __future__ import annotations

# Each method module must expose: build(df_models: pd.DataFrame, **params) -> pd.DataFrame



METHODS = {
    "A_baseline_all": ("selection.methods.A_baseline", {}),
    "B_longest_k5":   ("selection.methods.B_longest", {"k": 5}),
    "B_longest_k10":  ("selection.methods.B_longest", {"k": 10}),
    "B_longest_k20":  ("selection.methods.B_longest", {"k": 20}),
    "B_medium_k5":    ("selection.methods.B_medium", {"k": 5}),
    "B_medium_k10":   ("selection.methods.B_medium", {"k": 10}),
    "B_medium_k20":   ("selection.methods.B_medium", {"k": 20}),
    "B_balanced_k5":  ("selection.methods.B_rating_balanced", {"k_each": 2}),
    "B_balanced_k10": ("selection.methods.B_rating_balanced", {"k_each": 3}),
    "B_balanced_k20": ("selection.methods.B_rating_balanced", {"k_each": 6}),
    "C_best_sent_n10":("selection.methods.C_best_sentences", {"n": 10}),
    "C_best_sent_n15":("selection.methods.C_best_sentences", {"n": 15}),
    "C_best_sent_n20":("selection.methods.C_best_sentences", {"n": 20}),
    "D_tfidf_80":     ("selection.methods.D_tfidf_budget", {"budget": 80}),
    "D_tfidf_120":    ("selection.methods.D_tfidf_budget", {"budget": 120}),
    "D_tfidf_160":    ("selection.methods.D_tfidf_budget", {"budget": 160}),
    # Optional:
    # "E_dense_extractive": ("selection.methods.E_dense_extractive", {"clusters": 8}),
    # "E_llm_qwen": ("selection.methods.E_llm_ollama", {"model": "qwen2.5:7b"}),
    }

def iter_methods():
    for name, (module_path, params) in METHODS.items():
        yield name, module_path, params
import os
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.bm25 import BM25Retriever


# -------------------------
# LEVEL 2 CATEGORY FUNCTION
# -------------------------

def get_level2_categories(place_id, df_eval):

    row = df_eval[df_eval["id"] == place_id]

    if row.empty:
        return set()

    row = row.iloc[0]
    typeR = row["typeR"]

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

    return set(cat.strip() for cat in categories if cat.strip())


# -------------------------
# LOAD DATA ONCE
# -------------------------

print("Loading data...")

reviews = pd.read_csv("data/reviews83325.csv", low_memory=False)
places  = pd.read_csv("data/Tripadvisor.csv", low_memory=False)

reviews_en = reviews[reviews["langue"].str.lower() == "en"].copy()

df_models = reviews_en[["idplace", "review", "note"]].copy()
df_models.columns = ["place_id", "review_text", "score"]

df_eval = places[places["id"].isin(df_models["place_id"].unique())].copy()

print("Data loaded.")


# -------------------------
# DISCOVER SELECTION METHODS
# -------------------------

methods_path = "selection/methods"
method_files = [
    f.replace(".py", "")
    for f in os.listdir(methods_path)
    if f.endswith(".py") and not f.startswith("__")
]

print("\nDetected selection methods:")
for m in method_files:
    print(" -", m)


# -------------------------
# RUN EXPERIMENT FOR EACH
# -------------------------

results = []

for method_name in method_files:

    print("\n===================================")
    print("Running:", method_name)

    module = importlib.import_module(f"selection.methods.{method_name}")

    # Build corpus
    df_corpus = module.build(df_models)

    # Train/Test split
    train_ids, test_ids = train_test_split(
        df_corpus["place_id"],
        test_size=0.5,
        random_state=42
    )

    df_train = df_corpus[df_corpus["place_id"].isin(train_ids)]
    df_test  = df_corpus[df_corpus["place_id"].isin(test_ids)]

    # BM25
    retriever = BM25Retriever()
    retriever.fit(df_test)

    type_dict = dict(zip(df_eval["id"], df_eval["typeR"]))

    errors_l1 = []
    errors_l2 = []

    for i in range(len(df_train)):

        query_id = df_train.iloc[i]["place_id"]
        query_text = df_train.iloc[i]["text"]

        ranked_ids = retriever.rank(query_text)

        # -------- LEVEL 1 --------
        query_type = type_dict.get(query_id)

        for rank, candidate_id in enumerate(ranked_ids):
            if type_dict.get(candidate_id) == query_type:
                errors_l1.append(rank)
                break

        # -------- LEVEL 2 --------
        query_categories = get_level2_categories(query_id, df_eval)

        if not query_categories:
            continue

        for rank, candidate_id in enumerate(ranked_ids):
            candidate_categories = get_level2_categories(candidate_id, df_eval)

            if query_categories & candidate_categories:
                errors_l2.append(rank)
                break

    avg_l1 = np.mean(errors_l1)
    avg_l2 = np.mean(errors_l2)

    results.append({
        "method": method_name,
        "ranking_error_l1": avg_l1,
        "ranking_error_l2": avg_l2
    })

    print("Level 1:", avg_l1)
    print("Level 2:", avg_l2)


# -------------------------
# FINAL RESULTS TABLE
# -------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ranking_error_l1")

print("\n===================================")
print("FINAL COMPARISON")
print(results_df)
print("===================================")
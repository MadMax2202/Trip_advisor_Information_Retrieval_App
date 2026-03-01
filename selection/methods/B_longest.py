import pandas as pd

def build(df_models: pd.DataFrame, k: int = 10, **params) -> pd.DataFrame:
    tmp = df_models.copy()
    tmp["len"] = tmp["review_text"].astype(str).str.len()

    tmp = (
        tmp.sort_values(["place_id", "len"], ascending=[True, False])
           .groupby("place_id")
           .head(k)
    )

    out = (
        tmp.groupby("place_id", as_index=False)
           .agg(
               text=("review_text", lambda s: " ".join(s.tolist())),
               n_reviews_used=("review_text", "size"),
               avg_rating_used=("score", "mean"),
           )
    )
    return out
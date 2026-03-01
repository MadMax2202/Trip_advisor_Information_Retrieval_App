import pandas as pd

def build(df_models: pd.DataFrame, k: int = 10, **params) -> pd.DataFrame:
    tmp = df_models.copy()
    tmp["len"] = tmp["review_text"].astype(str).str.len()

    # Per-place median review length
    med = tmp.groupby("place_id")["len"].median().rename("med_len")
    tmp = tmp.join(med, on="place_id")

    # Rank reviews by distance to the median (closest = "medium")
    tmp["dist"] = (tmp["len"] - tmp["med_len"]).abs()

    tmp = (
        tmp.sort_values(["place_id", "dist"], ascending=[True, True])
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
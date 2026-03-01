import pandas as pd

def build(df_models: pd.DataFrame, k_each: int = 3, **params) -> pd.DataFrame:
    tmp = df_models.copy()

    # If score is missing or all NaN, fallback to longest reviews
    if "score" not in tmp.columns or tmp["score"].isna().all():
        tmp["len"] = tmp["review_text"].astype(str).str.len()
        tmp = (
            tmp.sort_values(["place_id", "len"], ascending=[True, False])
               .groupby("place_id")
               .head(k_each * 3)
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

    # Bucket ratings: low (<=2), mid (=3), high (>=4)
    def bucket(r):
        r = float(r)
        if r <= 2:
            return "low"
        elif r == 3:
            return "mid"
        return "high"

    tmp["bucket"] = tmp["score"].apply(bucket)
    tmp["len"] = tmp["review_text"].astype(str).str.len()

    # Take k_each reviews from each bucket per place (prefer longer ones)
    tmp = (
        tmp.sort_values(["place_id", "bucket", "len"], ascending=[True, True, False])
           .groupby(["place_id", "bucket"])
           .head(k_each)
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
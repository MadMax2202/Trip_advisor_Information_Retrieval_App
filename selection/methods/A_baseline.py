import pandas as pd

def build(df_models: pd.DataFrame, **params) -> pd.DataFrame:
    out = (
        df_models.groupby("place_id", as_index=False)
        .agg(
            text=("review_text", lambda s: " ".join(s.tolist())),
            n_reviews_used=("review_text", "size"),
            avg_rating_used=("score", "mean"),
        )
    )
    return out
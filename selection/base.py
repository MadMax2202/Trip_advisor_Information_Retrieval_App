from __future__ import annotations
import json
import pandas as pd

REQUIRED_OUT_COLS = ["place_id", "text"]
OPTIONAL_OUT_COLS = ["method", "params", "n_reviews_used", "avg_rating_used", "n_chars"]

def enforce_output_schema(out: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_OUT_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Selection method output missing columns: {missing}")

    out = out.copy()
    out["place_id"] = pd.to_numeric(out["place_id"], errors="coerce")
    out = out.dropna(subset=["place_id"])
    out["place_id"] = out["place_id"].astype(int)

    out["text"] = out["text"].fillna("").astype(str)
    out["n_chars"] = out["text"].str.len()

    # Keep consistent column ordering
    cols = REQUIRED_OUT_COLS + [c for c in OPTIONAL_OUT_COLS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra]

def params_to_str(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)
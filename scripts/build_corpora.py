from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib
import pandas as pd
from tqdm import tqdm

from selection.base import enforce_output_schema, params_to_str
from selection.cache import load_or_build
from selection.registry import iter_methods

CACHE_DIR = "data/processed_data/selection_cache"


def main():
    df_models = pd.read_parquet("data/processed_data/df_models.parquet")
    print("df_models:", df_models.shape)

    built = []
    for name, module_path, params in tqdm(list(iter_methods()), desc="Building corpora"):
        def _build():
            mod = importlib.import_module(module_path)
            out = mod.build(df_models, **params)

            out = enforce_output_schema(out)
            out["method"] = name
            out["params"] = params_to_str(params)

            if "n_reviews_used" not in out.columns:
                out["n_reviews_used"] = None
            if "avg_rating_used" not in out.columns:
                out["avg_rating_used"] = None

            out["n_chars"] = out["text"].astype(str).str.len()
            return out

        out, meta, hit = load_or_build(CACHE_DIR, name, params, _build)
        built.append((name, hit, out.shape[0], int(out["n_chars"].mean())))

    summary = pd.DataFrame(built, columns=["method", "cache_hit", "n_places", "avg_chars"])
    print("\n=== Summary ===")
    print(summary.sort_values("method").to_string(index=False))


if __name__ == "__main__":
    main()
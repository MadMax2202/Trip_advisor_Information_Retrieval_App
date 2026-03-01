from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

def _hash_key(method_name: str, params: dict) -> str:
    raw = json.dumps({"method": method_name, "params": params}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def cache_paths(cache_dir: str, method_name: str, params: dict):
    root = Path(cache_dir)
    (root / "corpora").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    key = _hash_key(method_name, params)
    data_path = root / "corpora" / f"{method_name}__{key}.parquet"
    meta_path = root / "meta" / f"{method_name}__{key}.json"
    return data_path, meta_path

def load_or_build(cache_dir: str, method_name: str, params: dict, build_fn):
    data_path, meta_path = cache_paths(cache_dir, method_name, params)
    if data_path.exists() and meta_path.exists():
        return pd.read_parquet(data_path), json.loads(meta_path.read_text(encoding="utf-8")), True

    out = build_fn()
    out.to_parquet(data_path, index=False)

    meta = {
        "method": method_name,
        "params": params,
        "rows": int(out.shape[0]),
        "cols": list(out.columns),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, meta, False
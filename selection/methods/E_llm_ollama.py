import time
from pathlib import Path
import pandas as pd
import requests


def _ollama_generate(model: str, prompt: str, timeout: int = 240) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def build(
    df_models: pd.DataFrame,
    model: str = "qwen2.5:3b",
    max_chars_input: int = 12000,
    limit_places: int | None = 50,
    sleep_s: float = 0.0,
    cache_root: str = "data/processed_data/selection_cache/ollama_place_cache",
    **params
) -> pd.DataFrame:
    """
    LLM summarization per place with PER-PLACE caching.

    Cache path:
      {cache_root}/{model}/{place_id}.txt

    limit_places:
      - use 50 for test
      - set to None for full run
    """
    cache_dir = Path(cache_root) / model.replace(":", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (place_id, sub) in enumerate(df_models.groupby("place_id")):
        if limit_places is not None and i >= limit_places:
            break

        place_id = int(place_id)
        cache_file = cache_dir / f"{place_id}.txt"

        if cache_file.exists():
            summary = cache_file.read_text(encoding="utf-8").strip()
        else:
            full_text = " ".join(sub["review_text"].astype(str).tolist())[:max_chars_input]

            prompt = (
                "You are summarizing TripAdvisor reviews into a compact retrieval-friendly description.\n"
                "Write ONE short paragraph (max ~120 words). Focus on distinctive aspects, atmosphere, "
                "service, value/price, location, and common pros/cons. Do NOT invent facts.\n\n"
                f"REVIEWS:\n{full_text}\n\nSUMMARY:"
            )

            tries = 3
            summary = ""
            for t in range(tries):
                try:
                    summary = _ollama_generate(model=model, prompt=prompt, timeout=240)
                    break
                except Exception:
                    if t == tries - 1:
                        summary = ""
                    else:
                        time.sleep(2.0 * (t + 1))

            cache_file.write_text(summary, encoding="utf-8")

            if sleep_s > 0:
                time.sleep(sleep_s)

        rows.append((place_id, summary, sub.shape[0], float(sub["score"].mean())))

    out = pd.DataFrame(rows, columns=["place_id", "text", "n_reviews_used", "avg_rating_used"])
    return out
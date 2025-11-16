from typing import List
import time
import random

import google.generativeai as genai

from src.app.settings import get_settings
import sys

# ------------------------------------------------------------------
# Configure Gemini
# ------------------------------------------------------------------
_settings = get_settings()
if not _settings.GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")

genai.configure(api_key=_settings.GOOGLE_API_KEY)


def _canon_model(name: str) -> str:
    """
    Ensure the embedding model name has the required 'models/' prefix.
    Falls back to text-embedding-004 if nothing is configured.
    """
    if not name:
        return "models/text-embedding-004"
    return name if name.startswith(("models/", "tunedModels/")) else f"models/{name}"


def _embed_one(text: str, model: str, attempt: int = 1, max_retries: int = 3) -> List[float]:
    """
    Call Gemini once for a single text with simple retry logic.
    This helps with transient 5xx errors from the API.
    """
    try:
        resp = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document",
        )
        return resp["embedding"]
    except Exception as e:
        if attempt >= max_retries:
            # Final failure – surface a clear error up to the caller
            print(f"[embed] FAILED after {attempt} attempts: {e}", flush=True)
            raise RuntimeError(f"Gemini embedding failed after {attempt} attempts: {e}") from e

        # Exponential-ish backoff with a bit of jitter
        sleep_for = 1.5 * attempt + random.random()
        print(
            f"[embed] Error on attempt {attempt}: {e} -> retrying in {sleep_for:.1f}s",
            flush=True,
        )
        time.sleep(sleep_for)
        return _embed_one(text, model, attempt + 1, max_retries=max_retries)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Gemini.

    - Logs progress to stdout (visible in Render logs).
    - Retries transient API errors instead of failing ingestion immediately.
    """
    model_name = getattr(_settings, "GEMINI_EMBED_MODEL", None)
    model = _canon_model(model_name)

    print(f"[embed] Model: {model} | texts={len(texts)}", flush=True)

    vectors: List[List[float]] = []
    total = len(texts)

    for i, t in enumerate(texts, 1):
        preview = (t[:60] + "…") if len(t) > 60 else t
        print(f"[embed] {i}/{total} -> “{preview}”", flush=True)

        emb = _embed_one(t, model)
        vectors.append(emb)

        print(f"[embed] done {i}, dim={len(emb)}", flush=True)

    print("[embed] all done", flush=True)
    return vectors

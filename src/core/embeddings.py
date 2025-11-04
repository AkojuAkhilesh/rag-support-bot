from typing import List
import google.generativeai as genai
from src.app.settings import get_settings
import sys

_settings = get_settings()
if not _settings.GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")

genai.configure(api_key=_settings.GOOGLE_API_KEY)

def _canon_model(name: str) -> str:
    if not name:
        return "models/text-embedding-004"
    return name if name.startswith(("models/", "tunedModels/")) else f"models/{name}"

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _canon_model(_settings.GEMINI_EMBED_MODEL)
    print(f"[embed] Model: {model} | texts={len(texts)}", flush=True)
    vectors: List[List[float]] = []
    for i, t in enumerate(texts, 1):
        preview = (t[:60] + "…") if len(t) > 60 else t
        print(f"[embed] {i}/{len(texts)} -> “{preview}”", flush=True)
        resp = genai.embed_content(model=model, content=t, task_type="retrieval_document")
        vectors.append(resp["embedding"])
        # show a small confirmation
        print(f"[embed] done {i}, dim={len(resp['embedding'])}", flush=True)
    print("[embed] all done", flush=True)
    return vectors

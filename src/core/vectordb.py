# src/core/vectordb.py
from typing import List, Dict
import os, uuid, pickle, math
from pathlib import Path
from src.core.embeddings import embed_texts
from src.app.settings import get_settings

_settings = get_settings()
INDEX_PATH = Path(".miniindex.pkl")

def _load_index():
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)  # dict with keys: texts, metas, vectors
    return {"texts": [], "metas": [], "vectors": []}

def _save_index(idx):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(idx, f)

def _cosine(a, b):
    # a,b: lists of floats
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na * nb)

def add_texts(texts: List[str], metadatas: List[Dict]):
    if len(texts) != len(metadatas):
        raise ValueError("documents and metadatas length mismatch")
    print(f"[mini-vs] embedding {len(texts)} text(s)...", flush=True)
    vecs = embed_texts(texts)
    idx = _load_index()
    idx["texts"].extend(texts)
    idx["metas"].extend(metadatas)
    idx["vectors"].extend(vecs)
    _save_index(idx)
    print(f"[mini-vs] ✅ saved. total={len(idx['texts'])}", flush=True)

def similarity_search(query: str, k: int = 6):
    idx = _load_index()
    if not idx["texts"]:
        print("[mini-vs] (empty index)", flush=True)
        return []
    qvec = embed_texts([query])[0]
    scores = [ _cosine(qvec, v) for v in idx["vectors"] ]
    # sort by highest cosine
    ranked = sorted(
        zip(scores, idx["texts"], idx["metas"]),
        key=lambda x: x[0],
        reverse=True
    )[:k]
    out = []
    for s, text, meta in ranked:
        out.append({"text": text, "meta": meta, "score": float(1.0 - s)})  # lower is better if you like; keep as-is
    print(f"[mini-vs] retrieved {len(out)} result(s)", flush=True)
    return out
# --- helpers for UI ---

def index_count() -> int:
    """Return number of stored chunks."""
    idx = _load_index()
    return len(idx["texts"])

def reset_index():
    """Delete the mini index file (fresh start)."""
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
# --- helpers for UI & summaries ---
from collections import Counter
from src.core.chunking import simple_chunk  # reuse your chunker

def index_count() -> int:
    """Total stored chunks."""
    idx = _load_index()
    return len(idx["texts"])

def reset_index():
    """Delete the mini index file (fresh start)."""
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

def index_summary() -> dict:
    """Summary: total chunks + counts per source."""
    idx = _load_index()
    counts = Counter([m.get("source", "?") for m in idx["metas"]])
    return {"total_chunks": len(idx["texts"]), "by_source": dict(counts)}

def add_document_text(name: str, text: str, chunk_size: int = 800, overlap: int = 120) -> int:
    """Chunk a raw text and add to index. Returns number of chunks added."""
    chunks = simple_chunk(text, chunk_size=chunk_size, overlap=overlap)
    metas = [{"source": name, "path": f"uploaded://{name}"} for _ in chunks]
    add_texts(chunks, metas)
    return len(chunks)

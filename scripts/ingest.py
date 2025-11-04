from pathlib import Path
from src.loaders.files import load_paths
from src.core.chunking import simple_chunk
from src.core.vectordb import add_texts
from src.app.settings import get_settings

def run_ingest(path: str):
    settings = get_settings()
    print(f"[ingest] starting, path={path}")  # <-- add
    root = Path(path)
    docs = load_paths(root)
    total_chunks = 0
    for d in docs:
        chunks = simple_chunk(d["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        metas = [{"source": d["meta"]["source"], "path": d["meta"]["path"]} for _ in chunks]
        add_texts(chunks, metas)
        total_chunks += len(chunks)
    print(f"[ingest] done, docs={len(docs)}, chunks={total_chunks}")  # <-- add


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw")
    args = ap.parse_args()
    run_ingest(args.path)
    print("✅ Ingestion completed successfully!")


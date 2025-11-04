from pathlib import Path
from typing import List, Dict

def load_paths(root: Path) -> List[Dict]:
    """
    Reads all .txt files under 'root' and returns a list of:
    {"text": "...", "meta": {"source": "faq.txt", "path": "data/raw/faq.txt"}}
    """
    docs: List[Dict] = []
    for p in root.rglob("*"):
        if p.suffix.lower() == ".txt":
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "meta": {"source": p.name, "path": str(p)}})
    return docs

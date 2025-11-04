# src/core/history.py
from pathlib import Path
import json, time
from typing import List, Dict, Optional

HIST_DIR = Path("data/history")
HIST_DIR.mkdir(parents=True, exist_ok=True)
HIST_FILE = HIST_DIR / "chat_history.jsonl"

def load_history() -> List[Dict]:
    """Load all messages from disk (JSONL)."""
    if not HIST_FILE.exists():
        return []
    out = []
    with open(HIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def append_message(role: str, content: str, citations: Optional[List[Dict]] = None):
    """Append a single message to history."""
    rec = {"ts": int(time.time()), "role": role, "content": content}
    if citations:
        rec["citations"] = citations
    with open(HIST_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def clear_history():
    """Delete the history file."""
    if HIST_FILE.exists():
        HIST_FILE.unlink()

def export_path() -> str:
    return str(HIST_FILE)

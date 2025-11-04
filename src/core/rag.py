from typing import Dict, List
import google.generativeai as genai
from src.core.vectordb import similarity_search
from src.app.settings import get_settings

settings = get_settings()

# --- Configure Gemini once ---
if not settings.GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")
genai.configure(api_key=settings.GOOGLE_API_KEY)

def _build_model():
    """
    Try the name as-is (preferred), then fall back to 'models/<name>' if needed.
    This handles SDK differences gracefully.
    """
    name = settings.GEMINI_MODEL or "gemini-1.5-flash"
    try:
        return genai.GenerativeModel(name)
    except Exception:
        alt = name if name.startswith(("models/", "tunedModels/")) else f"models/{name}"
        return genai.GenerativeModel(alt)

_model = _build_model()

def answer_query(query: str, top_k: int = 6) -> Dict:
    # 1) retrieve
    hits = similarity_search(query, k=top_k)
    if not hits:
        return {"answer": "I don’t know based on our docs.", "citations": []}

    # 2) build prompt
    context = "\n\n---\n\n".join([h["text"] for h in hits])
    prompt = f"""
You are a precise support assistant. Answer ONLY using the CONTEXT below.
If the answer is not in the context, say: "I don't know based on our docs."

CONTEXT:
{context}

USER QUESTION:
{query}

RESPONSE RULES:
- Be concise (2–5 sentences).
- At the end, add 'Sources:' and list [1], [2], ... based on the order of retrieved chunks.
"""

    # 3) generate (catch and surface any LLM errors so we don't 500)
    try:
        resp = _model.generate_content(prompt)
        answer = resp.text.strip() if hasattr(resp, "text") else ""
        if not answer:
            answer = "I don’t know based on our docs."
    except Exception as e:
        # Log to console and return a friendly message instead of 500
        print(f"[chat] LLM error: {e}", flush=True)
        answer = f"Sorry — the LLM call failed: {e}"

    # 4) citations
    citations: List[Dict] = []
    for i, h in enumerate(hits, 1):
        citations.append({
            "index": i,
            "source": h["meta"].get("source"),
            "path": h["meta"].get("path"),
            "score": h.get("score"),
        })
    return {"answer": answer, "citations": citations}

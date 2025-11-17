# src/app/main.py
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.rag import answer_query
from src.core.vectordb import add_texts, index_count, reset_index

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="RAG Support Bot (Gemini)")

# --- CORS so Streamlit on another origin can call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ok for demo; lock down later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Pydantic models ----------

# ----------------------------------------------------
# Pydantic models
# ----------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 6


class Citation(BaseModel):
    index: int
    source: Optional[str] = None
    path: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

class IngestRequest(BaseModel):
    texts: List[str]
    metas: List[Dict[str, Any]] = []

class IngestResponse(BaseModel):
    added: int
    total: int

# --------- Health ----------

@app.get("/")
def root():
    return {"message": "RAG API is running", "endpoints": ["/health", "/chat", "/ingest"]}

@app.get("/health")
def health():
    """Health check used by you and Render."""
    return {"status": "ok"}

# --------- Chat ----------

# ----------------------------------------------------
# Chat endpoint
# ----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    out = answer_query(req.query, top_k=req.top_k or 6)
    return ChatResponse(answer=out["answer"], citations=out["citations"])

# --------- Ingest (called from Streamlit) ----------

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Add new texts + metadata to the vector index.
    This runs INSIDE the rag-api service, so the index file we update
    is the same one used by /chat.
    """
    if not req.texts:
        return IngestResponse(added=0, total=index_count())

    add_texts(req.texts, req.metas or [{} for _ in req.texts])
    total = index_count()
    return IngestResponse(added=len(req.texts), total=total)

# (optional) an endpoint to reset index from outside
@app.post("/reset_index")
def reset():
    reset_index()
    return {"status": "cleared", "total": index_count()}

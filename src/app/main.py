from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.core.rag import answer_query

app = FastAPI(title="RAG Support Bot (Gemini)")

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    out = answer_query(req.query, top_k=req.top_k or 6)
    return ChatResponse(answer=out["answer"], citations=out["citations"])

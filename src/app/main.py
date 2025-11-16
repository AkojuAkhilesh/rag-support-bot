# src/app/main.py

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.rag import answer_query


# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="RAG Support Bot (Gemini)")


# CORS – allow your Streamlit UI (and others) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ----------------------------------------------------
# Basic routes
# ----------------------------------------------------
@app.get("/")
def root():
    """Simple root endpoint so hitting the base URL doesn't give 404."""
    return {
        "message": "RAG API is running",
        "endpoints": ["/health", "/chat"],
    }


@app.get("/health")
def health():
    """Health check used by you and Render."""
    return {"status": "ok"}


# ----------------------------------------------------
# Chat endpoint
# ----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Calls the RAG pipeline and returns answer + citations.
    """
    try:
        result = answer_query(req.query, top_k=req.top_k or 6)
        answer = result.get("answer", "")
        citations_raw = result.get("citations", [])

        # Ensure citations match the ChatResponse model
        citations: List[Citation] = [
            Citation(
                index=c.get("index", i),
                source=c.get("source"),
                path=c.get("path"),
                score=c.get("score"),
            )
            for i, c in enumerate(citations_raw)
        ]

        return ChatResponse(answer=answer, citations=citations)

    except HTTPException:
        # Re-raise if something inside already raised an HTTPException
        raise
    except Exception as e:
        # Log to server logs (Render will show this)
        print("[/chat] ERROR:", repr(e))

        # Return a clean error to the client
        raise HTTPException(
            status_code=500,
            detail="Internal error while generating answer. "
                   "Please try again or check server logs.",
        )

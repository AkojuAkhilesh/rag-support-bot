# --- make project root importable ---
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]  # -> C:\Users\dell\rag-support-bot
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so 'src' is importable
from src.core.history import load_history, append_message, clear_history, export_path
# streamlit_app.py
import requests
import streamlit as st
from pathlib import Path
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

# API_URL = "http://localhost:8000/chat"  # FastAPI endpoint

st.set_page_config(page_title="RAG Support Bot (Gemini)", page_icon="🤖", layout="centered")
st.title("🤖 RAG Support Bot (Gemini)")
st.caption("Ask questions about your ingested docs. (faq.txt for now)")

# Small status line
index_exists = Path(".miniindex.pkl").exists()
st.sidebar.header("Status")
st.sidebar.write("Index file:", "✅ found" if index_exists else "❌ missing (.miniindex.pkl)")
st.sidebar.write("API endpoint:", API_URL)

# Chat history in session
# if "messages" not in st.session_state:
    # st.session_state.messages = []  # each item: {"role": "user"/"assistant", "content": "...", "citations": [...]}
# Chat history in session (persisted)
if "messages" not in st.session_state:
    stored = load_history()
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "citations": m.get("citations", [])}
        for m in stored
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("citations"):
            with st.expander("Sources"):
                for c in m["citations"]:
                    st.write(f"[{c['index']}] {c.get('source')}  \n`{c.get('path')}`  \n(score: {c.get('score')})")

# Input box (sticky at bottom)
prompt = st.chat_input("Type your question…")

if prompt:
    # show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # persist user message to disk
    append_message("user", prompt)


    # call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(API_URL, json={"query": prompt, "top_k": 4}, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer.")
                    citations = data.get("citations", [])
                else:
                    answer = f"Server error: HTTP {resp.status_code}"
                    citations = []
            except Exception as e:
                answer = f"Request failed: {e}"
                citations = []

            st.markdown(answer)
            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        st.write(f"[{c['index']}] {c.get('source')}  \n`{c.get('path')}`  \n(score: {c.get('score')})")

            # persist assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            }

            )
            append_message("assistant", answer, citations)

# Sidebar controls
# st.sidebar.markdown("---")
# if st.sidebar.button("Clear chat"):
#     st.session_state.messages = []
#     st.rerun()
# Sidebar controls
st.sidebar.markdown("---")
with st.sidebar.expander("Conversation history"):
    # Download button
    try:
        with open(export_path(), "rb") as f:
            st.download_button("⬇️ Download history (.jsonl)", f, file_name="chat_history.jsonl")
    except Exception:
        st.caption("No history yet.")

    # Clear history + chat (confirm)
    confirm = st.checkbox("I understand this will erase the conversation")
    if st.button("Clear chat & history", disabled=not confirm):
        st.session_state.messages = []
        clear_history()
        st.sidebar.warning("History cleared.")
        st.rerun()

# import io
# from pypdf import PdfReader
# from src.core.vectordb import add_texts, index_count, reset_index
# from src.core.chunking import simple_chunk

# st.sidebar.markdown("### Upload & Ingest")
# uploaded = st.sidebar.file_uploader(
#     "Drop a .txt or .pdf file to add to the knowledge base",
#     type=["txt", "pdf"],
#     accept_multiple_files=False
# )

# if uploaded:
#     try:
#         if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
#             text = uploaded.read().decode("utf-8", errors="ignore")
#         else:
#             # PDF: extract text from all pages
#             pdf = PdfReader(io.BytesIO(uploaded.read()))
#             pages = []
#             for p in pdf.pages:
#                 pages.append(p.extract_text() or "")
#             text = "\n\n".join(pages)

#         # chunk it
#         chunks = simple_chunk(text, chunk_size=800, overlap=120)
#         metas = [{"source": uploaded.name, "path": f"uploaded://{uploaded.name}"} for _ in chunks]

#         # add to vector store
#         add_texts(chunks, metas)

#         st.sidebar.success(f"Ingested {len(chunks)} chunks from {uploaded.name}")
#         st.sidebar.write(f"Index size: **{index_count()}** chunks")

#     except Exception as e:
#         st.sidebar.error(f"Ingest failed: {e}")

# st.sidebar.markdown("---")
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     if st.button("Show index size"):
#         st.sidebar.info(f"Current chunks: **{index_count()}**")
# with col2:
#     if st.button("Reset index (danger)"):
#         reset_index()
#         st.sidebar.warning("Index cleared. Re-ingest files to use the bot.")
# ========= Upload & Index Controls =========
import io, os
from pathlib import Path
from typing import List
from pypdf import PdfReader
from src.core.vectordb import (
    add_document_text, index_count, reset_index, index_summary
)

st.sidebar.markdown("### Upload & Ingest")

# A) Single-file upload (txt/pdf)
uploaded = st.sidebar.file_uploader(
    "Drop a .txt or .pdf file",
    type=["txt", "pdf"],
    accept_multiple_files=False
)

def _read_file_bytes_to_text(uploaded_file) -> str:
    """Return full plain text for txt/pdf uploads."""
    if uploaded_file.type == "text/plain" or uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    # PDF
    pdf = PdfReader(io.BytesIO(uploaded_file.read()))
    pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages)

if uploaded:
    try:
        text = _read_file_bytes_to_text(uploaded)
        n = add_document_text(uploaded.name, text, chunk_size=800, overlap=120)
        st.sidebar.success(f"Ingested {n} chunks from {uploaded.name}")
        st.sidebar.write(f"Index size: **{index_count()}** chunks")
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")

# B) Multi-file upload (ingest several at once)
st.sidebar.markdown("#### Multi-file upload")
multi_files: List = st.sidebar.file_uploader(
    "Select multiple .txt/.pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)
if multi_files:
    total = 0
    prog = st.sidebar.progress(0.0, text="Ingesting…")
    for i, uf in enumerate(multi_files, 1):
        try:
            text = _read_file_bytes_to_text(uf)
            total += add_document_text(uf.name, text)
        except Exception as e:
            st.sidebar.error(f"{uf.name}: {e}")
        prog.progress(i / max(len(multi_files), 1))
    prog.empty()
    st.sidebar.success(f"Added {total} chunks from {len(multi_files)} file(s).")
    st.sidebar.write(f"Index size: **{index_count()}** chunks")

st.sidebar.markdown("---")

# C) Rebuild index from folder (data/raw)
st.sidebar.markdown("#### Rebuild from folder")
data_dir = Path("data/raw")
st.sidebar.caption(f"Folder: `{data_dir.as_posix()}`")
if st.sidebar.button("Rebuild index from data/raw (danger)"):
    try:
        reset_index()
        added = 0
        files = [p for p in data_dir.glob("**/*") if p.suffix.lower() in {".txt", ".pdf"}]
        for p in files:
            if p.suffix.lower() == ".txt":
                text = p.read_text(encoding="utf-8", errors="ignore")
            else:
                pdf = PdfReader(p.open("rb"))
                pages = [pg.extract_text() or "" for pg in pdf.pages]
                text = "\n\n".join(pages)
            added += add_document_text(p.name, text)
        st.sidebar.success(f"Rebuilt index: {added} chunks from {len(files)} file(s).")
        st.sidebar.write(f"Index size: **{index_count()}** chunks")
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

st.sidebar.markdown("---")

# D) Show what's indexed (summary)
if st.sidebar.button("Show indexed files"):
    summ = index_summary()
    st.sidebar.write(f"**Total chunks:** {summ['total_chunks']}")
    if summ["by_source"]:
        st.sidebar.write("**By source:**")
        for src, cnt in sorted(summ["by_source"].items(), key=lambda x: (-x[1], x[0])):
            st.sidebar.write(f"- {src}: {cnt}")
    else:
        st.sidebar.info("No documents indexed yet.")

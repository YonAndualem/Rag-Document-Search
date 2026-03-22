import logging
import os
import re
import tempfile
from datetime import datetime, timezone

import chromadb
import requests
import streamlit as st

from config import (
    CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, LLM_MODEL,
    MAX_QUERY_LENGTH, MAX_FILE_SIZE_MB, OLLAMA_BASE_URL,
    APP_PASSWORD, RATE_LIMIT_PER_MINUTE, MAX_HISTORY_TURNS
)
from logger import setup_logging
from rag_core import (
    extract_text_from_pdf, split_into_chunks,
    embed_chunks_concurrent, embed_text,
    build_messages, make_metadata, stream_answer
)

setup_logging()
logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORTED = True
except ImportError:
    DOCX_SUPPORTED = False

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Document Chat", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f8fafc, #eef2ff); }

    .main-title {
        font-size: 2.6rem; font-weight: 800; text-align: center;
        color: #111827; margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center; font-size: 1rem;
        color: #4b5563; margin-bottom: 1.5rem;
    }
    .onboarding-card {
        background: linear-gradient(135deg, #111827, #1e3a8a);
        padding: 2.5rem; border-radius: 22px; color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 1.5rem;
    }
    .onboarding-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.8rem; }
    .onboarding-step {
        background: rgba(255,255,255,0.1); border-radius: 12px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-size: 0.95rem;
    }
    .score-badge {
        display: inline-block; background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; border-radius: 8px;
        padding: 0.1rem 0.5rem; font-size: 0.8rem; font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .source-tag {
        color: #6b7280; font-size: 0.82rem; margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        return client.create_collection(name=COLLECTION_NAME)


def list_sources(collection) -> list:
    all_docs = collection.get(include=["metadatas"])
    sources = {}
    for meta in (all_docs.get("metadatas") or []):
        if meta and "source" in meta:
            sources[meta["source"]] = meta.get("uploaded_at", "")
    return sorted(sources.items())


def delete_source(collection, source: str):
    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Deleted %d chunks for: %s", len(existing["ids"]), source)


def check_ollama():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        missing = [m for m in (EMBED_MODEL, LLM_MODEL) if not any(m in n for n in models)]
        return missing if missing else True
    except requests.exceptions.ConnectionError:
        return None


def validate_file(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
    header = uploaded_file.read(4)
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".pdf") and header != PDF_MAGIC:
        return "Invalid PDF file."
    if name.endswith(".docx") and header != DOCX_MAGIC:
        return "Invalid DOCX file."
    return None


def extract_text(uploaded_file, temp_path: str) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(temp_path)
    if uploaded_file.name.lower().endswith(".docx") and DOCX_SUPPORTED:
        from docx import Document as DocxDocument
        doc = DocxDocument(temp_path)
        return " ".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def get_page_count(uploaded_file, temp_path: str) -> int:
    if uploaded_file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return len(PdfReader(temp_path).pages)
    return 0

# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = not APP_PASSWORD
if "request_times" not in st.session_state:
    st.session_state.request_times = []

# ── Auth gate ─────────────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    st.markdown('<div class="main-title">AI Document Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Enter the password to continue.</div>', unsafe_allow_html=True)
    pwd = st.text_input("Password", type="password")
    if st.button("Sign in", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### AI Document Chat")
    st.markdown("---")

    accept = ["pdf", "docx"] if DOCX_SUPPORTED else ["pdf"]
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=accept,
        help=f"PDF{' or DOCX' if DOCX_SUPPORTED else ''} · Max {MAX_FILE_SIZE_MB}MB"
    )

    if uploaded_file:
        err = validate_file(uploaded_file)
        if err:
            st.error(err)
        else:
            if st.button("Process document", type="primary", use_container_width=True):
                source = uploaded_file.name
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source)[1]) as f:
                        f.write(uploaded_file.read())
                        temp_path = f.name

                    logger.info("Processing: %s", source)
                    text = extract_text(uploaded_file, temp_path)

                    if not text.strip():
                        st.error("No text could be extracted. The file may be scanned or protected.")
                    else:
                        page_count = get_page_count(uploaded_file, temp_path)
                        chunks = split_into_chunks(text)

                        progress_bar = st.progress(0, text="Embedding chunks…")
                        status = st.empty()

                        def on_progress(done, total):
                            progress_bar.progress(done / total,
                                text=f"Embedding chunk {done} of {total}…")
                            status.caption(f"{done}/{total} chunks embedded")

                        embeddings = embed_chunks_concurrent(chunks, progress_callback=on_progress)
                        progress_bar.empty()
                        status.empty()

                        uploaded_at = datetime.now(timezone.utc).isoformat()
                        metadatas = [make_metadata(source, uploaded_at, i) for i in range(len(chunks))]

                        collection = get_collection()
                        existing = collection.get(where={"source": source})
                        if existing["ids"]:
                            collection.delete(ids=existing["ids"])

                        collection.add(
                            documents=chunks,
                            embeddings=embeddings,
                            ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
                            metadatas=metadatas
                        )
                        logger.info("Stored %d chunks for %s", len(chunks), source)

                        info = f"**{source}** ready"
                        if page_count:
                            info += f" · {page_count} pages"
                        info += f" · {len(chunks)} chunks"
                        st.success(info)
                        st.rerun()

                except Exception as e:
                    logger.exception("Failed to process document")
                    st.error(f"Failed: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

    st.markdown("---")
    st.markdown("**Uploaded Documents**")
    collection = get_collection()
    sources = list_sources(collection)

    if not sources:
        st.caption("No documents yet.")
    else:
        for src, ts in sources:
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{src}**")
            if ts:
                c1.caption(ts[:19].replace("T", " "))
            if c2.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                delete_source(collection, src)
                st.session_state.messages = []
                st.rerun()

    if sources and st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: {LLM_MODEL}  \nEmbed: {EMBED_MODEL}")

# ── Ollama health check ───────────────────────────────────────────────────────

ollama_status = check_ollama()
if ollama_status is None:
    st.error(f"Cannot connect to Ollama at `{OLLAMA_BASE_URL}`. Run: `ollama serve`")
    st.stop()
elif isinstance(ollama_status, list):
    st.error(
        f"Missing Ollama models: {', '.join(ollama_status)}. "
        f"Run: `ollama pull {' && ollama pull '.join(ollama_status)}`"
    )
    st.stop()

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">AI Document Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Upload documents, ask questions, get grounded answers from a local LLM.</div>',
    unsafe_allow_html=True
)

collection = get_collection()
sources = list_sources(collection)

# ── Empty state ───────────────────────────────────────────────────────────────

if not sources:
    st.markdown("""
    <div class="onboarding-card">
        <div class="onboarding-title">Get started</div>
        <div class="onboarding-step">① Upload a PDF or DOCX using the sidebar</div>
        <div class="onboarding-step">② Click <strong>Process document</strong> — chunks are embedded locally</div>
        <div class="onboarding-step">③ Ask any question about your document in the chat</div>
        <div class="onboarding-step">④ Answers stream in real time, grounded in your content</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col_dl, col_ctx = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    "Download answer",
                    data=msg["content"],
                    file_name="answer.txt",
                    mime="text/plain",
                    key=f"dl_{id(msg)}",
                )
            if msg.get("context_chunks"):
                with col_ctx:
                    with st.expander("Retrieved context"):
                        for chunk, src, score in zip(
                            msg["context_chunks"],
                            msg.get("context_sources", []),
                            msg.get("context_scores", [])
                        ):
                            relevance = max(0.0, 1.0 - score) * 100
                            st.markdown(
                                f'<div class="score-badge">Relevance {relevance:.0f}%</div>'
                                f'<div class="source-tag">Source: {src}</div>',
                                unsafe_allow_html=True
                            )
                            st.write(chunk)
                            st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

query = st.chat_input(f"Ask a question about your documents… (max {MAX_QUERY_LENGTH} chars)")

if query:
    import time as _time
    now = _time.time()
    st.session_state.request_times = [t for t in st.session_state.request_times if now - t < 60]

    if len(st.session_state.request_times) >= RATE_LIMIT_PER_MINUTE:
        st.error(f"Rate limit reached ({RATE_LIMIT_PER_MINUTE} requests/min). Please wait a moment.")
        st.stop()

    st.session_state.request_times.append(now)

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        st.error(f"Query too long ({len(query)}/{MAX_QUERY_LENGTH} chars).")
        st.stop()
    if not re.search(r'\w', query):
        st.error("Query must contain at least one word.")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    logger.info("Query: %s", query)

    # Embed query (with cache)
    if query not in st.session_state.embedding_cache:
        st.session_state.embedding_cache[query] = embed_text(query)
    query_embedding = st.session_state.embedding_cache[query]

    # Retrieve
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    if not docs or not docs[0]:
        with st.chat_message("assistant"):
            st.warning("No relevant context found. Try rephrasing your question.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "No relevant context found. Try rephrasing your question."
        })
        logger.warning("No results for query: %s", query)
    else:
        context_chunks = docs[0]
        context_sources = [m.get("source", "unknown") if m else "unknown" for m in (metas[0] or [])]
        context_scores = distances[0] if distances else []
        context = "\n\n".join(context_chunks)

        # Build conversation history — cap at MAX_HISTORY_TURNS pairs to avoid
        # exceeding the model's context window on long conversations
        prior = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]
        max_msgs = MAX_HISTORY_TURNS * 2  # each turn = 1 user + 1 assistant msg
        history = prior[-max_msgs:] if len(prior) > max_msgs else prior
        messages = build_messages(context, query, history)

        # Stream answer
        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(messages))

            col_dl, col_ctx = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    "Download answer",
                    data=answer,
                    file_name="answer.txt",
                    mime="text/plain",
                    key=f"dl_new_{len(st.session_state.messages)}",
                )
            with col_ctx:
                with st.expander("Retrieved context"):
                    for chunk, src, score in zip(context_chunks, context_sources, context_scores):
                        relevance = max(0.0, 1.0 - score) * 100
                        st.markdown(
                            f'<div class="score-badge">Relevance {relevance:.0f}%</div>'
                            f'<div class="source-tag">Source: {src}</div>',
                            unsafe_allow_html=True
                        )
                        st.write(chunk)
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context_chunks": context_chunks,
            "context_sources": context_sources,
            "context_scores": context_scores,
        })

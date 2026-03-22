# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Ollama running locally with two models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Python dependencies via virtual environment (system Python is externally managed on Debian/Ubuntu):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# Web UI
streamlit run streamlit_app.py

# CLI (loads data/docs/, interactive Q&A with streaming)
python app.py

# Tests (no Ollama or ChromaDB needed)
python -m pytest tests/ -v
```

## Architecture

### Module responsibilities

| File | Role |
|---|---|
| `config.py` | Single source of truth — loads all settings from `.env` via `python-dotenv` |
| `rag_core.py` | Shared library: PDF extraction, sentence-aware chunking, concurrent embedding, prompt building, streaming and non-streaming LLM calls |
| `streamlit_app.py` | Web UI entry point — chat interface, upload, auth, rate limiting |
| `app.py` | CLI entry point — Ollama health check, loads `data/docs/`, calls embed_store + chat_rag |
| `embed_store.py` | `store_embeddings(documents, source)` — embeds and stores chunks with metadata, replaces existing chunks for same source |
| `load_documents.py` | Loads `.txt`, `.pdf`, `.docx` from a folder, returns list of `{id, text}` dicts |
| `chat_rag.py` | CLI Q&A loop — multi-turn with conversation history, streaming output to stdout |
| `logger.py` | `setup_logging()` — configures logging level from `LOG_LEVEL` env var |

### Two execution paths

**Web UI** (`streamlit_app.py`):
- Upload → validate (size + magic bytes) → extract text → sentence-chunk → concurrent embed → store in ChromaDB with metadata
- Query → embed (cached per session) → retrieve top-3 with distances → stream LLM answer → display with relevance scores

**CLI** (`app.py` → `load_documents` → `embed_store` → `chat_rag`):
- Loads all files from `data/docs/` at startup, embeds and stores them
- Interactive loop: embed query → retrieve → stream answer to stdout → maintain conversation history

### Key design decisions

- **Sentence-aware chunking**: `split_into_chunks()` splits on `.!?` boundaries, groups sentences up to `CHUNK_SIZE`, falls back to character chunking for sentences longer than the limit.
- **Persistent ChromaDB**: `PersistentClient(path=CHROMA_PATH)` — embeddings survive restarts. Per-document replacement: existing chunks for a `source` are deleted before re-embedding.
- **Metadata on every chunk**: `source` (filename), `uploaded_at` (UTC ISO), `chunk_index` stored in ChromaDB for attribution and per-document deletion.
- **Concurrent embedding**: `embed_chunks_concurrent()` uses `ThreadPoolExecutor(max_workers=4)` with an optional `progress_callback` for the UI progress bar.
- **Streaming**: `stream_answer()` is a generator yielding tokens from `ollama.chat(stream=True)`. Web UI uses `st.write_stream()`, CLI writes to `sys.stdout` directly.
- **Auth**: Optional — only active when `APP_PASSWORD` env var is non-empty. Uses `st.session_state.authenticated`.
- **Rate limiting**: Sliding 60-second window tracked in `st.session_state.request_times`.

### Configuration

All tunable values live in `.env` (gitignored) and are loaded once by `config.py`. `.env.example` is the committed template. Never hardcode model names, paths, or limits in module files — always import from `config`.

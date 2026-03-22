# AI Document Chat (RAG)

A local Retrieval-Augmented Generation system for uploading documents and having a streamed, multi-turn conversation about their contents — fully offline using Ollama.

![AI Document Chat Demo](assets/rag-demo.png)

---

## Features

- Upload PDF or DOCX documents (up to 50MB, magic-byte validated)
- Sentence-aware text chunking for better retrieval quality
- Local embeddings via `nomic-embed-text` (Ollama)
- Persistent vector store with ChromaDB (survives restarts)
- Multi-document support — each document stored independently, deletable from sidebar
- Streaming LLM answers (token-by-token) via `gemma3:4b` (Ollama)
- Multi-turn chat with conversation history passed to the LLM
- Similarity scores shown for each retrieved context chunk
- Concurrent embedding (4 workers) with live progress bar
- Query embedding cache — same query never re-embedded
- Download any answer as `answer.txt`
- Optional password auth gate and per-session rate limiting
- CLI pipeline for batch-loading a folder of documents

---

## Tech Stack

- Python 3.12+
- [Streamlit](https://streamlit.io)
- [Ollama](https://ollama.com) — local LLM and embedding inference
- [ChromaDB](https://www.trychroma.com) — persistent vector database
- [PyPDF](https://pypdf.readthedocs.io) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io) — DOCX text extraction

---

## Setup

### 1. Install Ollama and pull the required models

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env as needed — defaults work out of the box
```

---

## Run

### Web UI

```bash
streamlit run streamlit_app.py
```

### CLI (batch-loads `data/docs/` folder)

```bash
python app.py
```

The CLI supports interactive multi-turn Q&A with streaming output. Place `.pdf`, `.docx`, or `.txt` files in `data/docs/` before running.

### Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma3:4b` | Ollama model for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `MAX_QUERY_LENGTH` | `500` | Max query characters |
| `RATE_LIMIT_PER_MINUTE` | `20` | Max queries per session per minute |
| `APP_PASSWORD` | _(empty)_ | Set to enable password auth gate |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `CHROMA_PATH` | `./chroma_db` | Where ChromaDB persists data |

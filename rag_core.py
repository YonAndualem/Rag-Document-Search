"""
Shared RAG utilities used by all entry points (streamlit_app, embed_store, chat_rag).

Backends:
  - Embeddings: Cohere API if COHERE_API_KEY is set, else Ollama
  - LLM:        Groq API  if GROQ_API_KEY  is set, else Ollama
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import os

import ollama
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, LLM_MODEL, GROQ_MODEL

logger = logging.getLogger(__name__)


# ── Text processing ────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.replace("\n", " ") + " "
    text = " ".join(text.split())
    logger.debug("Extracted %d characters from %s", len(text), path)
    return text


def split_into_chunks(text: str) -> list:
    """Sentence-aware chunker. Groups sentences into chunks up to CHUNK_SIZE,
    with CHUNK_OVERLAP characters of overlap between adjacent chunks."""
    text = " ".join(text.split())
    if not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= CHUNK_SIZE:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > CHUNK_SIZE:
                start = 0
                while start < len(sentence):
                    chunks.append(sentence[start:start + CHUNK_SIZE])
                    start += CHUNK_SIZE - CHUNK_OVERLAP
                current = ""
            else:
                overlap = chunks[-1][-CHUNK_OVERLAP:] if chunks else ""
                current = (overlap + " " + sentence).strip() if overlap else sentence

    if current.strip():
        chunks.append(current)

    result = [c for c in chunks if c.strip()]
    logger.debug("Split text into %d chunks", len(result))
    return result


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_text(text: str, input_type: str = "search_query") -> list:
    """Embed a single text string. input_type: 'search_query' or 'search_document'."""
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        import cohere
        co = cohere.ClientV2(api_key=cohere_key)
        resp = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type=input_type,
            embedding_types=["float"]
        )
        return list(resp.embeddings.float_[0])
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def embed_chunks_concurrent(chunks: list, max_workers: int = 4, progress_callback=None) -> list:
    """Embed all chunks. Uses Cohere batch API (1 call) or concurrent Ollama calls."""
    logger.info("Embedding %d chunks", len(chunks))
    t0 = time.time()

    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        import cohere
        co = cohere.ClientV2(api_key=cohere_key)
        resp = co.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"]
        )
        results = [list(e) for e in resp.embeddings.float_]
        if progress_callback:
            progress_callback(len(chunks), len(chunks))
    else:
        results = [None] * len(chunks)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(embed_text, chunk, "search_document"): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(chunks))

    logger.info("Embedded %d chunks in %.2fs", len(chunks), time.time() - t0)
    return results


# ── Prompt building ────────────────────────────────────────────────────────────

def make_metadata(source: str, uploaded_at: str, chunk_index: int) -> dict:
    return {"source": source, "uploaded_at": uploaded_at, "chunk_index": chunk_index}


def build_messages(context: str, query: str, history: list = None) -> list:
    """Build the messages list for the LLM, injecting context and conversation history."""
    system_content = (
        "You are a helpful AI assistant. "
        "Answer questions using only the context below. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def build_prompt(context: str, query: str) -> str:
    """Single-turn prompt string (used by CLI path)."""
    return (
        "You are a helpful AI assistant.\n"
        "Answer the user's question using only the context below.\n"
        "Keep the answer clear, concise, and professional.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )


# ── LLM inference ──────────────────────────────────────────────────────────────

def stream_answer(messages: list) -> Generator[str, None, None]:
    """Generator that streams tokens from the LLM (Groq or Ollama)."""
    t0 = time.time()

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        from groq import Groq
        logger.info("Streaming from Groq (%s)", GROQ_MODEL)
        client = Groq(api_key=groq_key)
        stream = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
    else:
        logger.info("Streaming from Ollama (%s)", LLM_MODEL)
        stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    logger.info("LLM stream finished in %.2fs", time.time() - t0)


def generate_answer(prompt: str) -> str:
    """Non-streaming answer (used by CLI path)."""
    t0 = time.time()

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        from groq import Groq
        logger.info("Generating answer via Groq (%s)", GROQ_MODEL)
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
    else:
        logger.info("Generating answer via Ollama (%s)", LLM_MODEL)
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"]

    logger.info("LLM responded in %.2fs", time.time() - t0)
    return result

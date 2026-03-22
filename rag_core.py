"""
Shared RAG utilities used by all entry points (streamlit_app, embed_store, chat_rag).
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)


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
                # Sentence longer than chunk size — fall back to character chunking
                start = 0
                while start < len(sentence):
                    chunks.append(sentence[start:start + CHUNK_SIZE])
                    start += CHUNK_SIZE - CHUNK_OVERLAP
                current = ""
            else:
                # Overlap: seed next chunk with end of previous
                overlap = chunks[-1][-CHUNK_OVERLAP:] if chunks else ""
                current = (overlap + " " + sentence).strip() if overlap else sentence

    if current.strip():
        chunks.append(current)

    result = [c for c in chunks if c.strip()]
    logger.debug("Split text into %d chunks", len(result))
    return result


def embed_text(text: str) -> list:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def embed_chunks_concurrent(chunks: list, max_workers: int = 4, progress_callback=None) -> list:
    """Embed all chunks in parallel. Calls progress_callback(done, total) after each chunk."""
    logger.info("Embedding %d chunks with %d workers", len(chunks), max_workers)
    t0 = time.time()
    results = [None] * len(chunks)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_text, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, len(chunks))

    elapsed = time.time() - t0
    logger.info("Embedded %d chunks in %.2fs", len(chunks), elapsed)
    return results


def make_metadata(source: str, uploaded_at: str, chunk_index: int) -> dict:
    return {"source": source, "uploaded_at": uploaded_at, "chunk_index": chunk_index}


def build_messages(context: str, query: str, history: list = None) -> list:
    """Build the messages list for ollama.chat, injecting context and conversation history."""
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


def stream_answer(messages: list):
    """Generator that streams tokens from the LLM."""
    logger.info("Streaming response from LLM (%s)", LLM_MODEL)
    t0 = time.time()
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
    logger.info("LLM stream finished in %.2fs", time.time() - t0)


def generate_answer(prompt: str) -> str:
    """Non-streaming answer (used by CLI path)."""
    logger.info("Sending prompt to LLM (%s)", LLM_MODEL)
    t0 = time.time()
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    logger.info("LLM responded in %.2fs", time.time() - t0)
    return response["message"]["content"]

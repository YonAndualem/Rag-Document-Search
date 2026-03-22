import os
import sys

import requests

from config import OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL
from logger import setup_logging

setup_logging()
from load_documents import load_documents
from embed_store import store_embeddings
from chat_rag import ask_rag

folder = "data/docs"


def check_ollama():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        missing = [m for m in (EMBED_MODEL, LLM_MODEL) if not any(m in name for name in models)]
        if missing:
            print(f"Error: Required Ollama models not found: {', '.join(missing)}")
            print(f"Run: ollama pull {' && ollama pull '.join(missing)}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {OLLAMA_BASE_URL}. Run: ollama serve")
        sys.exit(1)


check_ollama()

if not os.path.isdir(folder):
    os.makedirs(folder)
    print(f"Created '{folder}'. Add .pdf, .docx, or .txt files there and re-run.")
    sys.exit(0)

documents = load_documents(folder)
if not documents:
    print(f"No supported documents found in '{folder}'. Add .pdf, .docx, or .txt files and re-run.")
    sys.exit(0)

collection = store_embeddings(documents)
ask_rag(collection)

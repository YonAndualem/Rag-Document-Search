import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
APP_PASSWORD = os.getenv("APP_PASSWORD", "")          # empty = no auth
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 20))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 10))  # user+assistant pairs

import logging
import os

from rag_core import extract_text_from_pdf, split_into_chunks

logger = logging.getLogger(__name__)

DOCX_AVAILABLE = True
try:
    from docx import Document as DocxDocument
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed. DOCX files will be skipped.")


def extract_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    return " ".join(para.text for para in doc.paragraphs if para.text.strip())


def load_documents(folder: str) -> list:
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        text = None

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info("Loaded text file: %s", file)

        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
            logger.info("Loaded PDF: %s", file)

        elif file.endswith(".docx") and DOCX_AVAILABLE:
            text = extract_text_from_docx(path)
            logger.info("Loaded DOCX: %s", file)

        else:
            logger.debug("Skipping unsupported file: %s", file)
            continue

        if not text or not text.strip():
            logger.warning("No text extracted from %s, skipping", file)
            continue

        chunks = split_into_chunks(text)
        for i, chunk in enumerate(chunks):
            documents.append({"id": f"{file}_chunk_{i}", "text": chunk})

    logger.info("Loaded %d chunks from %s", len(documents), folder)
    return documents

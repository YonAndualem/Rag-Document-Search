# AI Document Search System (RAG)

## Demo

![AI Document Chat Demo](assets/rag-demo.png)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about them.

The system retrieves relevant document sections using embeddings and a vector database, then generates answers using a local language model.

---

## Features

- Upload PDF documents
- Document text extraction
- Text chunking for better retrieval
- Local embeddings using Ollama
- Vector database using ChromaDB
- Semantic search
- AI-generated answers using a local LLM
- Interactive Streamlit web interface

---

## Tech Stack

- Python
- Streamlit
- Ollama
- ChromaDB
- PyPDF
- Local LLM (Gemma)
- Embedding model (nomic-embed-text)

---

## How It Works

1. Upload a PDF document
2. Extract text from the document
3. Split the text into chunks
4. Convert chunks into embeddings
5. Store embeddings in ChromaDB
6. Convert user question into embedding
7. Retrieve the most relevant chunks
8. Generate the final answer with the LLM

---

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
import logging
import sys

from config import MAX_QUERY_LENGTH
from rag_core import embed_text, build_messages, stream_answer

logger = logging.getLogger(__name__)


def sanitize_query(query: str):
    query = query.strip()
    if not query:
        return None, "Query cannot be empty."
    if len(query) > MAX_QUERY_LENGTH:
        return None, f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
    return query, None


def ask_rag(collection):
    history = []   # [{role, content}] conversation memory

    while True:
        raw_query = input("Ask a question (or type 'exit'): ")

        if raw_query.strip().lower() == "exit":
            print("Goodbye!")
            break

        query, error = sanitize_query(raw_query)
        if error:
            print(f"Error: {error}\n")
            continue

        logger.info("Query received: %s", query)

        query_embedding = embed_text(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            print("No relevant context found for your query.\n")
            logger.warning("No results for query: %s", query)
            continue

        context = "\n\n".join(docs[0])
        print("\nBest Matching Context:\n")
        print(context[:600])
        print()

        messages = build_messages(context, query, history)

        print("Answer:\n")
        answer_parts = []
        for token in stream_answer(messages):
            sys.stdout.write(token)
            sys.stdout.flush()
            answer_parts.append(token)
        print("\n" + "-" * 50 + "\n")

        full_answer = "".join(answer_parts)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full_answer})

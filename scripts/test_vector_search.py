"""Smoke test for the embedding + pgvector search path, independent of
Ollama. Run this after scripts/embed_and_store.py to confirm retrieval
is wired up correctly before testing the full chatbot in main.py.

Usage:
    python scripts/test_vector_search.py "What is the minimum attendance requirement?"

If no argument is given, a default sample question is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_settings  # noqa: E402
from rag.embeddings import get_embedding_model  # noqa: E402
from rag.retriever import Retriever, RetrieverError  # noqa: E402

DEFAULT_QUESTION = "What is the minimum attendance requirement?"


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    settings = load_settings()

    print(f"Query: {question!r}\n")

    embedder = get_embedding_model(settings.embedding_model)
    vector = embedder.encode(question)
    print(f"Embedded to a {len(vector)}-dim vector (expect 384).\n")

    try:
        retriever = Retriever(settings.database_url)
    except RetrieverError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    try:
        results = retriever.search(vector, top_k=settings.top_k)
    finally:
        retriever.close()

    if not results:
        print(
            "No results. Either the knowledge base is empty "
            "(run scripts/embed_and_store.py) or nothing matched."
        )
        return

    print(f"Top {len(results)} result(s):\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. distance={r.distance:.4f}  title={r.title!r}  url={r.url}")
        preview = r.text[:200] + ("..." if len(r.text) > 200 else "")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()

"""Terminal RAG chatbot for IIT Bhubaneswar public information.

Usage:
    python main.py

Type a question, get an answer grounded in the ingested documents plus
their sources. Type `exit` or `quit` to leave.
"""

from __future__ import annotations

import sys

import requests

from rag.answer import AnswerGenerationError, generate_answer
from rag.config import load_settings
from rag.embeddings import get_embedding_model
from rag.retriever import Retriever, RetrieverError

EXIT_COMMANDS = {"exit", "quit", ":q"}


def check_ollama(ollama_host: str, model: str) -> None:
    """Fail fast with a clear message if Ollama isn't up, rather than
    letting the first question time out confusingly."""
    try:
        resp = requests.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(
            f"[startup error] Could not reach Ollama at '{ollama_host}'.\n"
            f"  Start it with `ollama serve` and make sure `{model}` is "
            f"pulled (`ollama pull {model}`).\n  Details: {exc}"
        )
        sys.exit(1)

    tags = [m["name"] for m in resp.json().get("models", [])]
    if tags and not any(model == t or model == t.split(":")[0] for t in tags):
        print(
            f"[warning] '{model}' was not found in `ollama list` "
            f"(available: {', '.join(tags) or 'none'}). The first request "
            f"may trigger a download or fail -- run `ollama pull {model}` "
            "if needed."
        )


def format_sources(chunks) -> str:
    if not chunks:
        return "  (no sources retrieved)"
    lines = []
    seen = set()
    for c in chunks:
        key = (c.title, c.url)
        if key in seen:
            continue
        seen.add(key)
        label = c.title or c.doc_id
        if c.url:
            lines.append(f"  - {label} ({c.url})")
        else:
            lines.append(f"  - {label}")
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()

    print("Loading embedding model (this happens once)...")
    embedder = get_embedding_model(settings.embedding_model)

    print("Connecting to PostgreSQL...")
    try:
        retriever = Retriever(settings.database_url)
    except RetrieverError as exc:
        print(f"[startup error] {exc}")
        sys.exit(1)

    print("Checking Ollama...")
    check_ollama(settings.ollama_host, settings.ollama_model)

    print(
        "\nIIT Bhubaneswar RAG Assistant -- ask a question "
        f"(type 'exit' to quit).\n"
    )

    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not question:
                continue
            if question.lower() in EXIT_COMMANDS:
                print("Exiting.")
                break

            query_vector = embedder.encode(question)

            try:
                chunks = retriever.search(query_vector, top_k=settings.top_k)
            except RetrieverError as exc:
                print(f"[retrieval error] {exc}\n")
                continue

            if not chunks:
                print(
                    "Assistant: I couldn't find this information in the "
                    "available college documents.\n"
                )
                continue

            try:
                answer = generate_answer(
                    question,
                    chunks,
                    model=settings.ollama_model,
                    ollama_host=settings.ollama_host,
                )
            except AnswerGenerationError as exc:
                print(f"[generation error] {exc}\n")
                continue

            print(f"\nAssistant: {answer}\n")
            print("Sources:")
            print(format_sources(chunks))
            print()
    finally:
        retriever.close()


if __name__ == "__main__":
    main()

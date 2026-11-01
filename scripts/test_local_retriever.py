"""End-to-end smoke test: retrieval + Ollama answer generation together,
using the three questions suggested in the project brief. Run this after
scripts/embed_and_store.py and with `ollama serve` running.

Usage:
    python scripts/test_local_retriever.py

This prints PASS/FAIL per question based on simple, honest checks (did
retrieval return anything, did generation succeed, did the model say it
couldn't find the answer for the known-unknown question). It does NOT
grade answer *quality* -- read the printed answers yourself and judge
that part, especially before writing test results into your README.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.answer import AnswerGenerationError, generate_answer  # noqa: E402
from rag.config import load_settings  # noqa: E402
from rag.embeddings import get_embedding_model  # noqa: E402
from rag.retriever import Retriever, RetrieverError  # noqa: E402

# Edit these to match documents you actually ingested. "Abhinav Arya" is
# a placeholder from the project brief -- replace it with a real name
# from your ingested faculty/notice pages before trusting the result.
TEST_CASES = [
    {
        "label": "In-knowledge-base question",
        "question": "Who is Abhinav Arya?",
    },
    {
        "label": "Multi-part follow-up",
        "question": "What is his research area, and which school does he belong to?",
    },
    {
        "label": "Known-unknown (should refuse, not fabricate)",
        "question": "What is his personal phone number?",
    },
]

REFUSAL_PHRASE = "couldn't find this information"


def main() -> None:
    settings = load_settings()

    print("Loading embedding model...")
    embedder = get_embedding_model(settings.embedding_model)

    print("Connecting to the database...")
    try:
        retriever = Retriever(settings.database_url)
    except RetrieverError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    results = []
    try:
        for case in TEST_CASES:
            print(f"\n{'=' * 60}\n{case['label']}\nQ: {case['question']}\n{'-' * 60}")

            vector = embedder.encode(case["question"])
            chunks = retriever.search(vector, top_k=settings.top_k)
            print(f"Retrieved {len(chunks)} chunk(s).")

            if not chunks:
                print("A: I couldn't find this information in the available college documents.")
                results.append((case["label"], "retrieval returned nothing"))
                continue

            try:
                answer = generate_answer(
                    case["question"],
                    chunks,
                    model=settings.ollama_model,
                    ollama_host=settings.ollama_host,
                )
            except AnswerGenerationError as exc:
                print(f"[generation error] {exc}")
                results.append((case["label"], "generation failed"))
                continue

            print(f"A: {answer}")

            if "phone number" in case["question"].lower():
                verdict = (
                    "OK: refused to fabricate"
                    if REFUSAL_PHRASE in answer.lower()
                    else "CHECK MANUALLY: did it invent a phone number?"
                )
            else:
                verdict = "generated an answer -- read it above and judge correctness yourself"
            results.append((case["label"], verdict))
    finally:
        retriever.close()

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for label, verdict in results:
        print(f"- {label}: {verdict}")

    print(
        "\nThis script does not auto-grade answer quality. Report in your "
        "README only what you actually observed when you ran this."
    )


if __name__ == "__main__":
    main()

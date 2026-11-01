"""Step 2 of ingestion: split documents into overlapping chunks.

Reads data/processed/documents.jsonl, splits each document's text into
word-based chunks with overlap, and writes data/processed/chunks.jsonl.

Usage:
    python scripts/chunk_documents.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "documents.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(words):
        chunk_words = words[start : start + size]
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
        start += step
    return chunks


def main() -> None:
    if not INPUT_FILE.exists():
        print(
            f"[error] {INPUT_FILE} not found. Run scripts/extract_pages.py "
            "first."
        )
        sys.exit(1)

    chunk_records = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            pieces = chunk_text(doc["text"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
            for idx, piece in enumerate(pieces):
                chunk_records.append(
                    {
                        "chunk_id": f"{doc['doc_id']}::{idx}",
                        "doc_id": doc["doc_id"],
                        "chunk_index": idx,
                        "title": doc.get("title"),
                        "url": doc.get("url"),
                        "text": piece,
                    }
                )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for rec in chunk_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunk_records)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

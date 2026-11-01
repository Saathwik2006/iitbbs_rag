"""Step 3 of ingestion: embed chunks and store them in pgvector.

Builds the SAME schema (kb_documents / kb_chunks) that the original
campus-rag-assistant project's LocalPgVectorRetriever provider
(backend/app/services/providers/retriever/local.py) queries against, so
this script and that provider can share one database if you want to.

Reads data/processed/documents.jsonl and data/processed/chunks.jsonl,
embeds each chunk with the model configured in .env, and upserts rows
(creating the tables and the pgvector extension if needed).

Usage:
    python scripts/embed_and_store.py

Safe to re-run: existing rows are upserted (ON CONFLICT DO UPDATE), so
this does not duplicate rows. Note this still recomputes embeddings for
every chunk on each run (the embedding model itself has no cache) -- for
a small corpus that's fine; see README "Known limitations" if that ever
becomes a bottleneck.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_settings  # noqa: E402
from rag.embeddings import EXPECTED_DIM, get_embedding_model  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DOCUMENTS_FILE = PROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS kb_documents (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    source_url  TEXT
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    metadata    JSONB,
    embedding   VECTOR({EXPECTED_DIM})
);
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO kb_documents (id, title, source_url)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    source_url = EXCLUDED.source_url;
"""

UPSERT_CHUNK_SQL = """
INSERT INTO kb_chunks (id, document_id, content, metadata, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding;
"""


def load_jsonl(path: Path, required_by: str) -> list[dict]:
    if not path.exists():
        print(f"[error] {path} not found. Run {required_by} first.")
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    settings = load_settings()
    documents = load_jsonl(DOCUMENTS_FILE, "scripts/extract_pages.py")
    chunks = load_jsonl(CHUNKS_FILE, "scripts/chunk_documents.py")

    if not chunks:
        print("[error] No chunks to embed -- chunks.jsonl is empty.")
        sys.exit(1)

    print(f"Loading embedding model '{settings.embedding_model}'...")
    embedder = get_embedding_model(settings.embedding_model)

    print("Connecting to PostgreSQL and preparing schema...")
    try:
        conn = psycopg.connect(settings.database_url, autocommit=True)
    except psycopg.OperationalError as exc:
        print(
            f"[error] Could not connect to '{settings.database_url}'. "
            f"Is PostgreSQL running? Details: {exc}"
        )
        sys.exit(1)

    with conn.cursor() as cur:
        try:
            cur.execute(CREATE_EXTENSION_SQL)
        except psycopg.errors.InsufficientPrivilege:
            print(
                "[error] Current DB user cannot CREATE EXTENSION vector. "
                "Ask a superuser to run `CREATE EXTENSION vector;` once "
                "on this database, then re-run this script."
            )
            sys.exit(1)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)

    print(f"Writing {len(documents)} documents to 'kb_documents'...")
    with conn.cursor() as cur:
        for doc in documents:
            cur.execute(
                UPSERT_DOCUMENT_SQL,
                (doc["doc_id"], doc.get("title"), doc.get("url")),
            )

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode_batch(texts)

    print("Writing to 'kb_chunks'...")
    with conn.cursor() as cur:
        for chunk, vector in tqdm(list(zip(chunks, vectors, strict=True))):
            metadata = {"chunk_index": chunk.get("chunk_index")}
            cur.execute(
                UPSERT_CHUNK_SQL,
                (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["text"],
                    json.dumps(metadata),
                    vector,
                ),
            )

    conn.close()
    print(f"\nDone. {len(chunks)} chunks across {len(documents)} documents are indexed.")


if __name__ == "__main__":
    main()

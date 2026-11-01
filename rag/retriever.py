"""PostgreSQL + pgvector retriever.

Schema matches the kb_documents / kb_chunks tables used by the
LocalPgVectorRetriever provider in the original campus-rag-assistant
project (backend/app/services/providers/retriever/local.py), so this
module works against a database already populated by that provider --
and scripts/embed_and_store.py in this project builds the same shape.
Does not perform any ingestion itself -- see scripts/ for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


@dataclass
class RetrievedChunk:
    text: str
    title: str | None
    url: str | None
    doc_id: str
    chunk_index: int | None
    distance: float


class RetrieverError(RuntimeError):
    """Raised when the database cannot be reached or queried."""


class Retriever:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        try:
            self._conn = psycopg.connect(database_url, autocommit=True)
            register_vector(self._conn)
        except psycopg.OperationalError as exc:
            raise RetrieverError(
                f"Could not connect to the database at '{database_url}'. "
                "Is PostgreSQL running and is the URL in your .env correct? "
                f"Original error: {exc}"
            ) from exc

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k chunks whose embedding is closest (cosine
        distance, via pgvector's <=> operator) to query_embedding.
        Returns an empty list if the table is empty or missing -- callers
        must handle that case explicitly rather than assuming results.
        """
        sql = """
            SELECT c.content, d.title, d.source_url, c.document_id,
                   c.chunk_index,
                   c.embedding <=> %s AS distance
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        try:
            with self._conn.cursor() as cur:
                query_vector = Vector(query_embedding)
                cur.execute(sql, (query_vector, query_vector, top_k))
                rows = cur.fetchall()
        except psycopg.errors.UndefinedTable as exc:
            raise RetrieverError(
                "The 'kb_chunks' / 'kb_documents' tables don't exist yet. "
                "Run scripts/embed_and_store.py to build the knowledge "
                "base before launching main.py."
            ) from exc

        return [
            RetrievedChunk(
                text=row[0],
                title=row[1],
                url=row[2],
                doc_id=row[3],
                chunk_index=row[4],
                distance=row[5],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

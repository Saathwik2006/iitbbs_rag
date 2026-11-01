"""Embedding model wrapper.

Loads sentence-transformers/all-MiniLM-L6-v2 (384-dim) exactly once and
reuses it for both document ingestion (scripts/embed_and_store.py) and
query-time embedding (rag/retriever.py via main.py), so query and
document vectors always come from the same model and preprocessing.
"""

from __future__ import annotations

from functools import lru_cache

EXPECTED_DIM = 384


class EmbeddingModel:
    """Thin, reusable wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so `python -c "import rag.embeddings"` and other
        # lightweight tooling don't pay the torch import cost unnecessarily.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        dim = self._model.get_sentence_embedding_dimension()
        if dim != EXPECTED_DIM:
            raise ValueError(
                f"Embedding model '{model_name}' produces {dim}-dim vectors, "
                f"but the database schema expects {EXPECTED_DIM}-dim vectors "
                "(see scripts/embed_and_store.py). Either use "
                "all-MiniLM-L6-v2 or update the schema and this constant."
            )

    def encode(self, text: str) -> list[float]:
        """Embed a single piece of text (e.g. a user question)."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed many chunks at once (used during ingestion)."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Process-wide singleton so the model is loaded once per run, not once
    per question. lru_cache keys on model_name so switching EMBEDDING_MODEL
    between runs still works correctly.
    """
    return EmbeddingModel(model_name)

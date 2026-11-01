"""Centralized configuration loading.

Every other module reads settings from here instead of calling
os.environ directly, so there is exactly one place that knows about
environment variable names and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str
    ollama_model: str
    ollama_host: str
    embedding_model: str
    top_k: int


def load_settings() -> Settings:
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://chatbot:chatbot@127.0.0.1:5432/chatbot_dev"
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k=int(os.getenv("TOP_K", "5")),
    )

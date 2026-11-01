"""Ollama-based answer generation.

Takes retrieved chunks + a question and asks the local llama3.2:3b model
(via Ollama) to answer strictly from that context. Contains no retrieval
logic -- see rag/retriever.py for that.
"""

from __future__ import annotations

import requests

from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a campus information assistant. Answer the user's question "
    "using ONLY the context provided below. The context comes from public "
    "IIT Bhubaneswar documents.\n\n"
    "Rules:\n"
    "- If the answer is not contained in the context, reply exactly: "
    "\"I couldn't find this information in the available college documents.\"\n"
    "- Never invent facts, names, phone numbers, URLs, or citations that "
    "are not present in the context.\n"
    "- Keep answers concise and directly address the question.\n"
    "- Do not mention that you were given 'context' -- just answer naturally."
)


class AnswerGenerationError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        context_block = "(no context was retrieved for this question)"
    else:
        parts = []
        for i, c in enumerate(chunks, start=1):
            label = c.title or c.doc_id
            parts.append(f"[Source {i}: {label}]\n{c.text}")
        context_block = "\n\n".join(parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT ---\n{context_block}\n--- END CONTEXT ---\n\n"
        f"Question: {question}\nAnswer:"
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    ollama_host: str,
    timeout: int = 120,
) -> str:
    """Call Ollama's /api/generate endpoint and return the model's answer
    text. Raises AnswerGenerationError if Ollama is unreachable.
    """
    prompt = _build_prompt(question, chunks)

    try:
        resp = requests.post(
            f"{ollama_host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise AnswerGenerationError(
            f"Could not reach Ollama at '{ollama_host}'. Is `ollama serve` "
            f"running and has `{model}` been pulled (`ollama pull {model}`)? "
            f"Original error: {exc}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise AnswerGenerationError(
            f"Ollama returned an error for model '{model}': {exc}. "
            f"Confirm the model is pulled with `ollama pull {model}`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise AnswerGenerationError(
            f"Ollama did not respond within {timeout}s. The model may still "
            "be loading into memory on first use -- try again."
        ) from exc

    data = resp.json()
    return data.get("response", "").strip()

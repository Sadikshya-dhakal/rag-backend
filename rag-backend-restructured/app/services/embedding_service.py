"""Embedding generation service.

Wraps `LLMClient.embed` behind a domain-specific name so callers (e.g.
`ingestion_service`) depend on "embed these chunks" rather than reaching
into the raw LLM client directly. Keeps a natural seam if embedding ever
needs its own caching, batching, or provider swap independent of chat.
"""
from __future__ import annotations

from app.llm.llm_client import LLMClient


def embed_texts(texts: list[str], llm: LLMClient) -> list[list[float]]:
    return llm.embed(texts)


def embed_query(query: str, llm: LLMClient) -> list[float]:
    return llm.embed_one(query)

"""Centralized application configuration.

All environment-dependent values are read once here via pydantic-settings
and reused across the app as a cached singleton (`get_settings`).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    chat_memory_ttl_seconds: int = 86400
    chat_memory_max_turns: int = 12

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # OpenAI-compatible LLM/embeddings
    openai_api_key: str = ""
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"

    # Chunking defaults
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120

    # Retrieval
    retrieval_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()

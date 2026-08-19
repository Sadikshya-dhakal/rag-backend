"""Reusable FastAPI dependency providers.

Client-style services (LLM, memory, vector store) are cached singletons;
repositories are constructed per-request from the request-scoped DB
session so they always see the current transaction.
"""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.llm.llm_client import LLMClient, get_llm_client
from app.memory.redis_memory import ChatMemory, get_chat_memory
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository
from app.vectorstore.qdrant_client import VectorStore, get_vector_store


def get_llm() -> LLMClient:
    return get_llm_client()


def get_memory() -> ChatMemory:
    return get_chat_memory()


def get_vectors() -> VectorStore:
    return get_vector_store()


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)


def get_booking_repository(db: Session = Depends(get_db)) -> BookingRepository:
    return BookingRepository(db)

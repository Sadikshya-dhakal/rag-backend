"""Qdrant vector store integration.

All Qdrant-specific code is isolated here behind a small, typed interface
(`VectorStore`) so the retrieval/RAG layer never touches the client
directly. This is the seam you'd change to swap in Milvus/Weaviate instead.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings


@dataclass(frozen=True)
class VectorRecord:
    vector_id: str
    vector: list[float]
    document_id: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    vector_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self._collection = settings.qdrant_collection
        self._dim = settings.embedding_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=self._dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        points = [
            qmodels.PointStruct(
                id=r.vector_id,
                vector=r.vector,
                payload={"document_id": r.document_id, "chunk_index": r.chunk_index, "text": r.text},
            )
            for r in records
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self, query_vector: list[float], top_k: int, document_id: str | None = None
    ) -> list[SearchHit]:
        query_filter = None
        if document_id:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            )
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            SearchHit(
                vector_id=str(hit.id),
                document_id=hit.payload["document_id"],
                chunk_index=hit.payload["chunk_index"],
                text=hit.payload["text"],
                score=hit.score,
            )
            for hit in results
        ]

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                )
            ),
        )


def new_vector_id() -> str:
    return str(uuid.uuid4())


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()

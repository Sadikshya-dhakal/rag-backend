"""Document ingestion pipeline: extract -> chunk -> embed -> store.

Orchestrates extraction, chunking, embedding, vector storage, and metadata
persistence (via `DocumentRepository`). The raw uploaded file is also saved
to `uploads/` for auditability/reprocessing.
"""
from __future__ import annotations

import os
import uuid

from fastapi import UploadFile

from app.core.config import get_settings
from app.llm.llm_client import LLMClient
from app.models.document import Document, DocumentChunk
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import ChunkingStrategy
from app.services.chunking_service import get_chunker
from app.services.embedding_service import embed_texts
from app.services.extraction import extract_text
from app.vectorstore.qdrant_client import VectorRecord, VectorStore, new_vector_id

UPLOAD_DIR = "uploads"


def _persist_raw_file(filename: str, raw_bytes: bytes) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{filename}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(raw_bytes)
    return path


async def ingest_document(
    *,
    file: UploadFile,
    strategy: ChunkingStrategy,
    chunk_size: int | None,
    chunk_overlap: int | None,
    repository: DocumentRepository,
    vector_store: VectorStore,
    llm: LLMClient,
) -> Document:
    settings = get_settings()
    resolved_chunk_size = chunk_size or settings.default_chunk_size
    resolved_chunk_overlap = chunk_overlap or settings.default_chunk_overlap

    raw_bytes = await file.read()
    await file.seek(0)
    text, file_type = await extract_text(file)
    _persist_raw_file(file.filename or "unnamed", raw_bytes)

    document = Document(
        filename=file.filename or "unnamed",
        file_type=file_type,
        chunking_strategy=strategy.value,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        char_count=len(text),
        status="processing",
    )
    repository.create(document)

    chunker = get_chunker(strategy, resolved_chunk_size, resolved_chunk_overlap)
    chunks = chunker.split(text)

    if chunks:
        embeddings = embed_texts([c.text for c in chunks], llm)
        vector_records: list[VectorRecord] = []
        chunk_rows: list[DocumentChunk] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_id = new_vector_id()
            vector_records.append(
                VectorRecord(
                    vector_id=vector_id,
                    vector=embedding,
                    document_id=document.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )
            )
            chunk_rows.append(
                DocumentChunk(
                    document_id=document.id,
                    vector_id=vector_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )
            )

        vector_store.upsert(vector_records)
        repository.add_chunks(chunk_rows)

    document.num_chunks = len(chunks)
    document.status = "ready"
    repository.commit()
    repository.refresh(document)
    return document

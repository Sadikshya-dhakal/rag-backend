"""Data-access layer for documents and document chunks.

Services depend on this repository rather than importing SQLAlchemy models
or writing queries directly — keeps persistence concerns out of business
logic and makes services easy to unit test with a fake repository.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, document: Document) -> Document:
        self._db.add(document)
        self._db.flush()  # assigns document.id without committing
        return document

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        self._db.add_all(chunks)

    def commit(self) -> None:
        self._db.commit()

    def refresh(self, document: Document) -> None:
        self._db.refresh(document)

    def get(self, document_id: str) -> Document | None:
        return self._db.get(Document, document_id)

    def list_all(self) -> list[Document]:
        return list(self._db.execute(select(Document).order_by(Document.created_at.desc())).scalars().all())

    def delete(self, document: Document) -> None:
        self._db.delete(document)
        self._db.commit()

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_document_repository, get_llm, get_vectors
from app.llm.llm_client import LLMClient
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    ChunkingStrategy,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.services.ingestion_service import ingest_document
from app.vectorstore.qdrant_client import VectorStore

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post("/upload", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="A .pdf or .txt file."),
    chunking_strategy: ChunkingStrategy = Form(default=ChunkingStrategy.RECURSIVE_PARAGRAPH),
    chunk_size: int | None = Form(default=None, ge=100, le=8000),
    chunk_overlap: int | None = Form(default=None, ge=0, le=2000),
    repository: DocumentRepository = Depends(get_document_repository),
    vector_store: VectorStore = Depends(get_vectors),
    llm: LLMClient = Depends(get_llm),
) -> Document:
    """Upload a document, chunk it with the selected strategy, embed the
    chunks, and persist both the vectors (Qdrant) and metadata (SQL)."""
    return await ingest_document(
        file=file,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        repository=repository,
        vector_store=vector_store,
        llm=llm,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(repository: DocumentRepository = Depends(get_document_repository)) -> DocumentListResponse:
    rows = repository.list_all()
    return DocumentListResponse(documents=[DocumentSummary.model_validate(r) for r in rows], total=len(rows))


@router.get("/{document_id}", response_model=DocumentIngestResponse)
def get_document(
    document_id: str, repository: DocumentRepository = Depends(get_document_repository)
) -> Document:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
    vector_store: VectorStore = Depends(get_vectors),
) -> None:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    vector_store.delete_document(document_id)
    repository.delete(document)

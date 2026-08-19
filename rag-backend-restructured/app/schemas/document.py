from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE_PARAGRAPH = "recursive_paragraph"


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    char_count: int
    status: str
    created_at: datetime


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    num_chunks: int
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class ChunkPreview(BaseModel):
    chunk_index: int
    text: str = Field(..., description="Chunk text, truncated for preview.")

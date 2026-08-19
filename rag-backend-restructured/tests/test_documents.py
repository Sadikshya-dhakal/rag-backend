"""Tests for chunking strategies and document ingestion.

Chunking tests are pure unit tests (no external services). The upload
endpoint test requires live Qdrant/Redis/OpenAI and is gated behind
RUN_INTEGRATION_TESTS=1.
"""
from __future__ import annotations

import os

import pytest

from app.services.chunking_service import FixedSizeChunker, RecursiveParagraphChunker


def test_fixed_size_chunker_basic_windowing() -> None:
    text = "a" * 1000
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text == text[0:200]
    assert chunks[1].text == text[150:350]


def test_fixed_size_chunker_empty_text() -> None:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("   ") == []


def test_fixed_size_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)


def test_recursive_paragraph_chunker_respects_paragraph_boundaries() -> None:
    text = "\n\n".join([f"Paragraph {i}. " + ("word " * 20) for i in range(10)])
    chunker = RecursiveParagraphChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c.text) <= 350 for c in chunks)
    reconstructed = " ".join(c.text for c in chunks)
    assert "Paragraph 0" in reconstructed
    assert "Paragraph 9" in reconstructed


def test_recursive_paragraph_chunker_handles_single_long_paragraph() -> None:
    text = "word " * 500
    chunker = RecursiveParagraphChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_indices_are_sequential() -> None:
    text = "\n\n".join([f"Para {i}" for i in range(5)])
    chunker = RecursiveParagraphChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.split(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires live Qdrant/Redis/OpenAI credentials; set RUN_INTEGRATION_TESTS=1 to run.",
)
def test_upload_txt_document() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    files = {"file": ("sample.txt", b"This is a sample document about FastAPI and RAG systems.", "text/plain")}
    data = {"chunking_strategy": "fixed_size", "chunk_size": "200", "chunk_overlap": "20"}
    response = client.post("/documents/upload", files=files, data=data)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["num_chunks"] >= 1

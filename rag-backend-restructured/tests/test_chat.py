"""Tests for the conversational RAG chat endpoint.

Requires live Qdrant/Redis/OpenAI credentials; gated behind
RUN_INTEGRATION_TESTS=1 since these hit real external services.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires live Qdrant/Redis/OpenAI credentials; set RUN_INTEGRATION_TESTS=1 to run.",
)


def test_chat_without_documents_returns_rag_answer() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/chat", json={"session_id": "test-chat-1", "message": "Hello, who are you?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "rag_answer"
    assert body["answer"]


def test_chat_history_round_trips() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    session_id = "test-chat-history-1"
    client.post("/chat", json={"session_id": session_id, "message": "What is FastAPI?"})

    history_response = client.get(f"/chat/{session_id}/history")
    assert history_response.status_code == 200
    turns = history_response.json()["turns"]
    assert len(turns) >= 2  # user + assistant

    clear_response = client.delete(f"/chat/{session_id}/history")
    assert clear_response.status_code == 204

    history_after_clear = client.get(f"/chat/{session_id}/history")
    assert history_after_clear.json()["turns"] == []


def test_multi_turn_follow_up_uses_history() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    session_id = "test-chat-multiturn-1"
    client.post("/chat", json={"session_id": session_id, "message": "What is Python?"})
    follow_up = client.post("/chat", json={"session_id": session_id, "message": "What about its typing system?"})
    assert follow_up.status_code == 200
    assert follow_up.json()["answer"]

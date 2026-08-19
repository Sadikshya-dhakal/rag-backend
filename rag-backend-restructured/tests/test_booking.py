"""Tests for the interview-booking flow: direct API and chat-driven slot filling.

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


def test_direct_booking_creation() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    payload = {
        "session_id": "test-booking-direct-1",
        "name": "Jane Doe",
        "email": "jane@example.com",
        "date": "2026-09-01",
        "time": "14:00",
    }
    response = client.post("/bookings", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Jane Doe"
    assert body["status"] == "confirmed"

    get_response = client.get(f"/bookings/{body['id']}")
    assert get_response.status_code == 200


def test_booking_via_chat_slot_filling() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    session_id = "test-booking-chat-1"

    r1 = client.post("/chat", json={"session_id": session_id, "message": "I'd like to book an interview"})
    assert r1.status_code == 200
    assert r1.json()["intent"] == "booking_in_progress"

    r2 = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "My name is John Smith, email john@example.com, on 2026-09-05 at 10:00",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["intent"] == "booking_confirmed"


def test_list_bookings() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.get("/bookings")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable client-supplied conversation id.")
    message: str = Field(..., min_length=1, description="User's latest message.")
    document_id: str | None = Field(
        default=None, description="Optional: restrict retrieval to a single document."
    )


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float


class BookingSlots(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    intent: str = Field(description="'rag_answer' | 'booking_in_progress' | 'booking_confirmed'")
    booking_slots: BookingSlots | None = None


class ChatHistoryTurn(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    turns: list[ChatHistoryTurn]

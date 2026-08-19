"""Interview-booking flow.

Uses the LLM's function-calling to (a) decide whether the user's message is
part of a booking flow, and (b) pull out whichever of {name, email, date,
time} are present in the message. Slots persist in Redis across turns until
all four are collected, then the booking is written to SQL via
`BookingRepository`.

No LangChain chains are used; this is a hand-rolled two-step LLM interaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.llm.llm_client import LLMClient
from app.memory.redis_memory import ChatMemory
from app.models.booking import InterviewBooking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import InterviewBookingCreate

_INTENT_TOOL_NAME = "handle_booking"
_INTENT_TOOL_DESCRIPTION = (
    "Call this if the user wants to schedule/book an interview, OR if the conversation "
    "is already mid-way through collecting booking details (name, email, date, time). "
    "Extract any of those four fields present in the latest message. Omit fields not mentioned."
)
_INTENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_booking_related": {
            "type": "boolean",
            "description": "True if this message is about scheduling an interview or supplies booking details.",
        },
        "name": {"type": "string", "description": "Candidate's full name, if mentioned."},
        "email": {"type": "string", "description": "Candidate's email address, if mentioned."},
        "date": {"type": "string", "description": "Interview date in YYYY-MM-DD format, if mentioned."},
        "time": {"type": "string", "description": "Interview time in HH:MM 24h format, if mentioned."},
    },
    "required": ["is_booking_related"],
}

_REQUIRED_SLOTS = ("name", "email", "date", "time")


@dataclass
class BookingTurnResult:
    is_booking_related: bool
    slots: dict[str, str | None] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    completed_booking: InterviewBooking | None = None


def _merge_slots(existing: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for slot in _REQUIRED_SLOTS:
        value = extracted.get(slot)
        if value:
            merged[slot] = value
    return merged


def process_booking_turn(
    *,
    session_id: str,
    user_message: str,
    history: list[dict[str, str]],
    llm: LLMClient,
    memory: ChatMemory,
    repository: BookingRepository,
) -> BookingTurnResult:
    """Runs one turn of the booking slot-filling flow.

    Returns `is_booking_related=False` if this message has nothing to do
    with booking, so the caller can fall back to a normal RAG answer.
    """
    system_prompt = (
        "You are an assistant that detects interview-booking intent and extracts "
        "booking details (name, email, date, time) from user messages. "
        "Only extract values explicitly present in the LATEST user message."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})

    extracted = llm.extract_structured(
        messages=messages,
        tool_name=_INTENT_TOOL_NAME,
        tool_description=_INTENT_TOOL_DESCRIPTION,
        parameters_schema=_INTENT_PARAMETERS,
    )

    existing_state = memory.get_booking_state(session_id)
    was_already_in_flow = bool(existing_state)

    if not extracted or not extracted.get("is_booking_related", False):
        if not was_already_in_flow:
            return BookingTurnResult(is_booking_related=False)
        extracted = extracted or {}

    merged = _merge_slots(existing_state, extracted)
    missing = [slot for slot in _REQUIRED_SLOTS if not merged.get(slot)]

    if missing:
        memory.set_booking_state(session_id, merged)
        return BookingTurnResult(is_booking_related=True, slots=merged, missing_slots=missing)

    booking_in = InterviewBookingCreate(
        session_id=session_id,
        name=merged["name"],
        email=merged["email"],
        date=merged["date"],
        time=merged["time"],
    )
    record = create_booking(repository, booking_in)
    memory.clear_booking_state(session_id)
    return BookingTurnResult(is_booking_related=True, slots=merged, missing_slots=[], completed_booking=record)


def create_booking(repository: BookingRepository, booking_in: InterviewBookingCreate) -> InterviewBooking:
    record = InterviewBooking(
        session_id=booking_in.session_id,
        name=booking_in.name,
        email=booking_in.email,
        interview_date=booking_in.date,
        interview_time=booking_in.time,
        status="confirmed",
    )
    return repository.create(record)

"""Per-turn chat orchestration.

Entry point called by the `/chat` route. Each turn:

    1. Checks booking intent first (`booking_service`) — if the message is
       booking-related or continues an in-progress booking, that flow
       handles the turn entirely and RAG is skipped.
    2. Otherwise, runs the custom retrieval + generation pipeline
       (`rag_service`) and returns a grounded answer.

Conversation history and booking slot state both live in Redis
(`redis_memory.ChatMemory`) so multi-turn context survives across requests.
"""
from __future__ import annotations

from app.llm.llm_client import LLMClient
from app.memory.redis_memory import ChatMemory
from app.repositories.booking_repository import BookingRepository
from app.schemas.chat import BookingSlots, ChatResponse
from app.services import rag_service
from app.services.booking_service import process_booking_turn
from app.vectorstore.qdrant_client import VectorStore


def handle_chat_turn(
    *,
    session_id: str,
    user_message: str,
    document_id: str | None,
    booking_repository: BookingRepository,
    memory: ChatMemory,
    vector_store: VectorStore,
    llm: LLMClient,
) -> ChatResponse:
    history = memory.get_history(session_id)

    # 1. Booking intent takes priority.
    booking_result = process_booking_turn(
        session_id=session_id,
        user_message=user_message,
        history=history,
        llm=llm,
        memory=memory,
        repository=booking_repository,
    )
    if booking_result.is_booking_related:
        memory.append_turn(session_id, "user", user_message)

        if booking_result.completed_booking:
            b = booking_result.completed_booking
            answer = (
                f"You're booked! Interview confirmed for {b.name} on {b.interview_date} "
                f"at {b.interview_time}. A confirmation will be sent to {b.email}."
            )
            memory.append_turn(session_id, "assistant", answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                sources=[],
                intent="booking_confirmed",
                booking_slots=BookingSlots(**{k: v for k, v in booking_result.slots.items()}),
            )

        missing = ", ".join(booking_result.missing_slots)
        answer = f"Sure — I can help book your interview. Could you share your {missing}?"
        memory.append_turn(session_id, "assistant", answer)
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=[],
            intent="booking_in_progress",
            booking_slots=BookingSlots(**{k: v for k, v in booking_result.slots.items()}),
        )

    # 2. Otherwise, run custom RAG.
    standalone_query = rag_service.rewrite_query(user_message, history, llm)
    retrieved = rag_service.retrieve(
        query=standalone_query, document_id=document_id, vector_store=vector_store, llm=llm
    )
    answer = rag_service.generate_answer(user_message=user_message, history=history, retrieved=retrieved, llm=llm)

    memory.append_turn(session_id, "user", user_message)
    memory.append_turn(session_id, "assistant", answer)

    return ChatResponse(session_id=session_id, answer=answer, sources=retrieved, intent="rag_answer", booking_slots=None)

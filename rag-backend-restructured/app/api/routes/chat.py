from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_booking_repository, get_llm, get_memory, get_vectors
from app.llm.llm_client import LLMClient
from app.memory.redis_memory import ChatMemory
from app.repositories.booking_repository import BookingRepository
from app.schemas.chat import ChatHistoryResponse, ChatHistoryTurn, ChatRequest, ChatResponse
from app.services.chat_service import handle_chat_turn
from app.vectorstore.qdrant_client import VectorStore

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    booking_repository: BookingRepository = Depends(get_booking_repository),
    memory: ChatMemory = Depends(get_memory),
    vector_store: VectorStore = Depends(get_vectors),
    llm: LLMClient = Depends(get_llm),
) -> ChatResponse:
    """Single endpoint handling multi-turn RAG *and* interview-booking flows.

    Routing is intent-based per turn: if the message is booking-related (or
    a booking is already in progress for this session), it's handled by the
    booking slot-filler; otherwise it goes through custom retrieval + a
    grounded LLM answer.
    """
    return handle_chat_turn(
        session_id=request.session_id,
        user_message=request.message,
        document_id=request.document_id,
        booking_repository=booking_repository,
        memory=memory,
        vector_store=vector_store,
        llm=llm,
    )


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
def get_history(session_id: str, memory: ChatMemory = Depends(get_memory)) -> ChatHistoryResponse:
    turns = memory.get_history(session_id)
    return ChatHistoryResponse(session_id=session_id, turns=[ChatHistoryTurn(**t) for t in turns])


@router.delete("/{session_id}/history", status_code=204, response_model=None)
def clear_history(session_id: str, memory: ChatMemory = Depends(get_memory)) -> None:
    memory.clear_history(session_id)
    memory.clear_booking_state(session_id)

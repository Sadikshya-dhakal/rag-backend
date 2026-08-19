"""Custom retrieval-augmented generation.

Deliberately hand-rolled (no `RetrievalQAChain` / LangChain chains):

    1. Reformulate the latest user message into a standalone query using
       recent chat history (handles multi-turn follow-ups like "what about
       its accuracy?").
    2. Embed the standalone query and retrieve top-k chunks from Qdrant.
    3. Build a grounded prompt from the retrieved chunks + conversation
       history and call the chat model directly.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.llm.llm_client import LLMClient
from app.schemas.chat import RetrievedChunk
from app.services.embedding_service import embed_query
from app.vectorstore.qdrant_client import VectorStore

_QUERY_REWRITE_SYSTEM = (
    "Rewrite the user's latest message as a standalone question that makes sense "
    "without the prior conversation. Preserve intent exactly. If it is already "
    "standalone, return it unchanged. Reply with ONLY the rewritten question, no preamble."
)

_RAG_SYSTEM_TEMPLATE = (
    "You are a helpful assistant answering questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't have enough information "
    "in the ingested documents — do not make things up.\n\n"
    "Context:\n{context}"
)


def rewrite_query(user_message: str, history: list[dict[str, str]], llm: LLMClient) -> str:
    if not history:
        return user_message
    messages = [{"role": "system", "content": _QUERY_REWRITE_SYSTEM}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})
    rewritten = llm.chat(messages, temperature=0)
    return rewritten or user_message


def _build_context(hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "(no relevant context retrieved)"
    return "\n\n".join(
        f"[Source {i + 1} | doc={h.document_id} chunk={h.chunk_index}]\n{h.text}" for i, h in enumerate(hits)
    )


def retrieve(
    *, query: str, document_id: str | None, vector_store: VectorStore, llm: LLMClient
) -> list[RetrievedChunk]:
    settings = get_settings()
    query_vector = embed_query(query, llm)
    hits = vector_store.search(query_vector, top_k=settings.retrieval_top_k, document_id=document_id)
    return [
        RetrievedChunk(document_id=h.document_id, chunk_index=h.chunk_index, text=h.text, score=h.score)
        for h in hits
    ]


def generate_answer(
    *, user_message: str, history: list[dict[str, str]], retrieved: list[RetrievedChunk], llm: LLMClient
) -> str:
    system_message = {"role": "system", "content": _RAG_SYSTEM_TEMPLATE.format(context=_build_context(retrieved))}
    messages = [system_message, *history[-6:], {"role": "user", "content": user_message}]
    return llm.chat(messages, temperature=0.2)

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import bookings, chat, documents
from app.core.database import init_db
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    yield


app = FastAPI(
    title="RAG Backend",
    description=(
        "Document ingestion + conversational RAG backend. "
        "Custom retrieval pipeline (no RetrievalQAChain), Qdrant vector store, "
        "Redis chat memory, LLM-driven interview booking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(bookings.router)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}

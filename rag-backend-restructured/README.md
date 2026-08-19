# RAG Backend — Document Ingestion + Conversational RAG

A modular FastAPI backend with two REST APIs:

1. **Document Ingestion API** — upload `.pdf`/`.txt`, chunk with a selectable strategy, embed, and store in Qdrant + SQL metadata.
2. **Conversational RAG API** — custom (hand-rolled, no `RetrievalQAChain`) multi-turn RAG over Redis-backed chat memory, with LLM-driven interview booking.

## Stack

| Concern            | Choice                                    |
|---------------------|--------------------------------------------|
| API framework        | FastAPI                                    |
| Vector store          | Qdrant (no FAISS/Chroma)                   |
| Metadata store         | SQLite by default, swappable to Postgres via `DATABASE_URL` (SQLAlchemy 2.0) |
| Chat memory           | Redis                                      |
| Embeddings / LLM        | OpenAI-compatible API (`OPENAI_BASE_URL` lets you point at any compatible provider) |
| PDF extraction          | pdfplumber                                 |

## Architecture

```
rag-backend/
├── app/
│   ├── main.py                     # FastAPI app, router registration, lifespan (DB init)
│   ├── api/routes/                 # documents.py, chat.py, bookings.py
│   ├── core/
│   │   ├── config.py                 # pydantic-settings, single source of truth for env vars
│   │   ├── database.py                # SQLAlchemy engine/session/Base
│   │   └── logging.py
│   ├── models/                     # ORM models
│   │   ├── document.py               # Document, DocumentChunk
│   │   └── booking.py                # InterviewBooking, ChatMessage
│   ├── schemas/                    # Pydantic request/response models
│   ├── repositories/               # DB access boundary used by services
│   │   ├── document_repository.py
│   │   └── booking_repository.py
│   ├── vectorstore/
│   │   └── qdrant_client.py          # Qdrant integration, isolated behind VectorStore
│   ├── memory/
│   │   └── redis_memory.py           # Redis-backed rolling chat history + booking slot state
│   ├── llm/
│   │   └── llm_client.py             # OpenAI-compatible client wrapper (chat + embeddings + tool calling)
│   └── services/
│       ├── ingestion_service.py       # orchestrates extract -> chunk -> embed -> store
│       ├── chunking_service.py        # FixedSizeChunker & RecursiveParagraphChunker
│       ├── embedding_service.py        # domain wrapper over llm_client's embed calls
│       ├── extraction.py               # PDF/TXT -> raw text
│       ├── rag_service.py              # custom retrieval pipeline (query rewrite -> retrieve -> generate)
│       ├── chat_service.py             # per-turn orchestration: booking vs. RAG
│       └── booking_service.py          # LLM-driven interview-booking slot extraction
├── tests/
│   ├── test_documents.py
│   ├── test_chat.py
│   └── test_booking.py
├── uploads/                       # raw uploaded files, persisted for audit/reprocessing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Each layer only talks to the layer directly below it through a narrow, typed interface — routes depend on repositories/services, services depend on repositories/vectorstore/memory/llm, and none of those leak SQLAlchemy/Qdrant/Redis/OpenAI specifics upward. This is what makes swapping Qdrant→Weaviate, or OpenAI→another provider, a one-file change.

## Why no `RetrievalQAChain`

`app/services/rag_service.py` + `app/services/chat_service.py` implement the RAG loop manually:

1. **Query rewrite** — the latest user message is rewritten into a standalone question using the last few turns of history (handles "what about its accuracy?"-style follow-ups).
2. **Retrieve** — the rewritten query is embedded and searched against Qdrant (optionally filtered to one `document_id`).
3. **Grounded generation** — retrieved chunks are formatted into a context block and passed to the chat model directly via `llm_client.chat(...)`, with the system prompt instructing it to answer only from context.

No LangChain chain abstractions are used anywhere in the pipeline.

## Chunking strategies

Both implement a shared `Chunker` protocol (`app/services/chunking_service.py`), selectable per-upload via the `chunking_strategy` form field:

- **`fixed_size`** — fixed-width character sliding window with configurable overlap. Predictable, fast, language-agnostic.
- **`recursive_paragraph`** — splits on paragraphs first, recursing into sentences then whitespace only for oversized pieces, then greedily packs small paragraphs together up to `chunk_size` with a small overlap tail carried forward. Produces chunks that respect natural text boundaries.

## Interview booking flow

`app/services/booking_service.py` uses the LLM's **function/tool calling** (not chains) to, on every chat turn:

1. Decide if the message is booking-related (or continues an in-progress booking).
2. Extract whichever of `{name, email, date, time}` are present in that message.
3. Merge newly extracted slots into the session's Redis-persisted booking state.
4. Once all four slots are filled, write an `InterviewBooking` row via `BookingRepository` and clear the Redis state.

This means booking can span multiple turns ("I'd like to book an interview" → "John Doe" → "john@x.com, next Tuesday at 3pm") and survives process restarts (state lives in Redis, not in-memory).

## API summary

### Document Ingestion
- `POST /documents/upload` — multipart form: `file`, `chunking_strategy` (`fixed_size`|`recursive_paragraph`), optional `chunk_size`, `chunk_overlap`
- `GET /documents` — list ingested documents
- `GET /documents/{id}` — document detail
- `DELETE /documents/{id}` — removes SQL metadata + vectors from Qdrant

### Conversational RAG
- `POST /chat` — body: `{ "session_id": str, "message": str, "document_id": str | null }`
- `GET /chat/{session_id}/history` — rolling Redis history for a session
- `DELETE /chat/{session_id}/history` — clear session memory + any in-progress booking state

### Bookings
- `POST /bookings` — direct booking creation (bypasses chat/LLM)
- `GET /bookings` / `GET /bookings/{id}`

Full interactive schema at `/docs` once running.

## Running locally (without Docker for the app)

```bash
# 1. Start Qdrant + Redis only
docker compose up -d qdrant redis

# 2. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# edit .env and set OPENAI_API_KEY (or point OPENAI_BASE_URL at another OpenAI-compatible provider)

# 4. Run
uvicorn app.main:app --reload
```

## Running fully in Docker

```bash
cp .env.example .env   # set OPENAI_API_KEY first
docker compose up --build
```

This builds the API image from the `Dockerfile` and starts it alongside Qdrant and Redis, all networked together.

Visit `http://localhost:8000/docs`.

### Example: ingest a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@sample.pdf" \
  -F "chunking_strategy=recursive_paragraph" \
  -F "chunk_size=800" \
  -F "chunk_overlap=120"
```

### Example: chat (RAG)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What does the document say about pricing?"}'
```

### Example: chat (booking)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "I would like to book an interview"}'
# -> asks for name, email, date, time

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "Jane Doe, jane@example.com, 2026-09-01 at 14:00"}'
# -> booking confirmed, row written to SQL
```

## Tests

```bash
pytest tests/ -v                            # pure unit tests (chunking), integration tests auto-skip
RUN_INTEGRATION_TESTS=1 pytest tests/ -v     # full endpoint tests, requires Qdrant/Redis/OpenAI
```

## Design notes / trade-offs

- **SQLite by default** for zero-setup evaluation; swapping to Postgres is a one-line `DATABASE_URL` change since everything goes through SQLAlchemy 2.0's typed ORM.
- **Repository layer** (`app/repositories/`) keeps services free of raw SQLAlchemy queries, so persistence logic can be swapped or mocked independently of business logic.
- **Embeddings are stored only in Qdrant**; SQL keeps a `DocumentChunk` row per chunk (with `vector_id` pointer + raw text) purely for auditability/debugging without needing to query the vector DB.
- **Redis is the source of truth for live conversation state** (both chat history and in-progress booking slots); SQL `ChatMessage` acts as a durable log — this mirrors how most production chat systems separate hot working-memory from cold storage.
- **`session_id` is client-supplied** (e.g. a UUID generated by the caller) rather than server-issued, keeping the API stateless-friendly and easy to test with `curl`.
- **Raw uploaded files are saved to `uploads/`** for auditability/reprocessing, separate from the extracted-text pipeline.

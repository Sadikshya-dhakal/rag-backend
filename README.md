# RAG Backend — Simple Setup Guide

This is a backend with two APIs:
1. Upload PDF/TXT documents and search them (RAG)
2. Chat with the documents, including booking an interview

Follow these steps in order.

---

## What you need installed first

- **Docker Desktop** — download from docker.com, install it, and make sure it's running.
- **Ollama** (free, runs AI models on your own computer, no account/card needed) — download from ollama.com.

Check both are installed by running in a terminal:
```
docker --version
ollama --version
```
Both should print a version number.

---

## Step 1: Download the AI models (one time only)

```
ollama pull llama3.1
ollama pull nomic-embed-text
```
This downloads a few GB. Wait for each to finish before running the next.

---

## Step 2: Set up your environment file

In the project folder, run:
```
cp .env.example .env
```

Open the new `.env` file in a text editor and make sure these lines look exactly like this:

```
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
CHAT_MODEL=llama3.1
```

Save the file.

> This points the project at your local Ollama instead of paid OpenAI — completely free, no API key needed.

---

## Step 3: Start everything

From inside the project folder, run:
```
docker compose up --build
```

Wait until you see this line with no red errors after it:
```
Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal window open — it's running your server. Don't close it.

---

## Step 4: Check it's working

Open a browser and go to:
```
http://localhost:8000/docs
```

You should see a page listing the API endpoints. If that loads, it's working.

---

## Step 5: Upload a document

**Option A — using the browser page:**
1. On the `/docs` page, find **POST /documents/upload**
2. Click it, then click "Try it out"
3. Choose your PDF or TXT file
4. Click "Execute"
5. Look for `"status": "ready"` in the response — that means it worked

**Option B — using the terminal:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@yourfile.pdf" \
  -F "chunking_strategy=recursive_paragraph"
```

This may take 10 seconds to a couple minutes since it's running on your own computer.

---

## Step 6: Chat with your document

Open a new terminal tab (leave the server one running) and run:
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"session_id\": \"test-1\", \"message\": \"What is this document about?\"}"
```

You should get back a JSON response with an `"answer"` field.

---

## Step 7: Book an interview through chat

**First message** (starts the booking):
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"session_id\": \"booking-1\", \"message\": \"I would like to book an interview\"}"
```

**Second message** (give your details, same session_id):
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"session_id\": \"booking-1\", \"message\": \"name: John Smith, email: john@example.com, date: 2026-09-05, time: 14:00\"}"
```

If it worked, the response will include `"intent": "booking_confirmed"`.

---

## Common problems

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker Desktop isn't installed or isn't running — open the Docker Desktop app |
| `no configuration file provided` | You're in the wrong folder — run `cd rag-backend-restructured` first |
| `insufficient_quota` error | You're still pointed at OpenAI, not Ollama — check Step 2 again |
| Upload takes a long time | Normal for local models — just wait, don't cancel |
| JSON errors when pasting into `/docs` | Clear the box completely (Ctrl+A, Delete) before pasting new text |

---

## Stopping the server

In the terminal running the server, press `Ctrl + C`.

To fully stop and remove containers:
```
docker compose down
```

To also wipe stored data and start completely fresh:
```
docker compose down -v
```

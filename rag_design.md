# RAG System — Simplified Overview

## What is RAG?
RAG (Retrieval-Augmented Generation) lets an LLM answer questions about **your own documents** by fetching relevant content before generating a response. It reduces hallucination by grounding answers in real retrieved text.

---

## Core Components

| Component | Technology | Purpose |
|---|---|---|
| Backend | FastAPI | API routes & orchestration |
| LLM | Ollama (local) | Answer generation |
| Embeddings | SentenceTransformers | Convert text → vectors |
| Vector DB | Pinecone | Similarity search |
| Document DB | SQLite | File storage & metadata |

---

## Two Main Pipelines

### 1. Upload (Ingestion)
```
File Upload → Parse → Chunk → Embed → Store
```
- **Parse** — extract text from PDF, TXT, or Image (OCR)
- **Chunk** — split into ~1000-char overlapping pieces
- **Embed** — convert each chunk to a vector (1536 dimensions)
- **Store** — vectors → Pinecone, file metadata → SQLite

### 2. Query (Retrieval + Generation)
```
Question → Embed → Search → Gate → Prompt → LLM → Answer
```
- **Embed** — convert question to a vector
- **Search** — find top-5 similar chunks in Pinecone
- **Gate** — if best score < 0.3, return "not enough info"
- **Prompt** — combine chunks + chat history + question
- **LLM** — Ollama generates the answer

---

## Key Design Choices

**Why chunk text?** Embedding models have token limits, and smaller chunks give more precise search results.

**Why overlap between chunks?** Prevents sentences from being cut off at boundaries.

**Why a confidence gate?** Stops the LLM from hallucinating when no relevant context is found.

**Why local LLM + embeddings?** Free, private, and no data leaves your machine.

**Why stateless server?** Chat history is sent by the client each request — no session management needed.

---

## Response Format
Every query returns:
- `answer` — LLM-generated response (or `null` if gated)
- `sources` — matched chunks with scores and snippets
- `confidence` — highest similarity score
- `message` — explanation if no answer was generated
# RAG Backend — API Documentation

Base URL: `http://localhost:8000`

Interactive Swagger docs: `http://localhost:8000/docs`

All endpoints are async. Errors return JSON with a `detail` field.

---

## Health Check

### `GET /health`

Check if the backend is operational.

**Response 200:**
```json
{
  "status": "healthy"
}
```

---

## Document Upload

### `POST /documents/upload`

Upload a document for ingestion into the knowledge base. The file is parsed, chunked, embedded, and stored in both SQLite (metadata + raw bytes) and Pinecone (vectors).

**Content-Type:** `multipart/form-data`

**Form Fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | Yes | — | The document file. Accepted types: PDF, TXT, PNG, JPEG |
| `source` | string | Yes | — | Label for the document (e.g., "Q4 Report") |
| `chunk_size` | integer | No | 1000 | Characters per chunk (100–4000) |
| `chunk_overlap` | integer | No | 200 | Overlap between adjacent chunks (0–999, must be < chunk_size) |

**Supported MIME Types:**
- `application/pdf`
- `text/plain`
- `image/png`
- `image/jpeg`

**Response 200:**
```json
{
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "report.pdf",
  "chunk_count": 42,
  "message": "Document ingested successfully"
}
```

**Error 422 — Unsupported file type:**
```json
{
  "detail": "Unsupported file type: application/zip. Supported: pdf, text, png, jpeg"
}
```

**Error 422 — File too large:**
```json
{
  "detail": "File size exceeds maximum allowed size of 52428800 bytes"
}
```

**Error 422 — Invalid chunk params:**
```json
{
  "detail": "chunk_overlap must be less than chunk_size"
}
```

**Error 422 — No extractable text:**
```json
{
  "detail": "No text could be extracted from the uploaded file"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf" \
  -F "source=Q4 Financial Report" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

---

## List Documents

### `GET /documents`

List all uploaded documents with metadata. Does not include file content.

**Response 200:**
```json
{
  "documents": [
    {
      "document_id": "a1b2c3d4-...",
      "filename": "report.pdf",
      "content_type": "application/pdf",
      "source": "Q4 Financial Report",
      "file_size": 245760,
      "uploaded_at": "2025-01-15T10:30:00+00:00"
    }
  ]
}
```

**Response 200 — Empty:**
```json
{
  "documents": []
}
```

Documents are ordered by upload time (newest first).

---

## Get Document Detail

### `GET /documents/{document_id}`

Get metadata for a single document, including chunk count.

**Path Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `document_id` | string | UUID of the document |

**Response 200:**
```json
{
  "document_id": "a1b2c3d4-...",
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "source": "Q4 Financial Report",
  "file_size": 245760,
  "uploaded_at": "2025-01-15T10:30:00+00:00",
  "chunk_count": 42
}
```

**Error 404:**
```json
{
  "detail": "Document not found"
}
```

---

## Download Document

### `GET /documents/{document_id}/download`

Download the original uploaded file.

**Path Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `document_id` | string | UUID of the document |

**Response 200:**
- Body: Raw file bytes
- `Content-Type`: Original MIME type (e.g., `application/pdf`)
- `Content-Disposition`: `attachment; filename="report.pdf"`

**Error 404:**
```json
{
  "detail": "Document not found"
}
```

---

## Delete Document

### `DELETE /documents/{document_id}`

Delete a document from both SQLite and Pinecone.

**Path Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `document_id` | string | UUID of the document |

**Response 200 — Full success:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "a1b2c3d4-..."
}
```

**Response 200 — Partial (Pinecone cleanup failed):**
```json
{
  "message": "Document record deleted but vector cleanup failed",
  "document_id": "a1b2c3d4-..."
}
```

**Error 404:**
```json
{
  "detail": "Document not found"
}
```

---

## Query Knowledge Base

### `POST /query`

Ask a question against the uploaded documents using the RAG pipeline.

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | Yes | — | The question to ask (1–2000 chars) |
| `top_k` | integer | No | 5 | Number of chunks to retrieve (1–20) |
| `filter` | object | No | null | Pinecone metadata filter |
| `chat_history` | array | No | null | Recent conversation messages for context |

**`chat_history` item schema:**

| Field | Type | Description |
|---|---|---|
| `role` | string | `"user"` or `"assistant"` |
| `content` | string | Message text |

**`filter` examples:**
```json
{"source": "Q4 Report"}
{"source": {"$in": ["Q4 Report", "Q3 Report"]}}
{"document_id": {"$eq": "a1b2c3d4-..."}}
```

**Response 200 — With answer:**
```json
{
  "answer": "The revenue in Q4 was $5.2M, representing a 15% increase...",
  "sources": [
    {
      "document_id": "a1b2c3d4-...",
      "source": "Q4 Financial Report",
      "chunk_index": 3,
      "score": 0.65,
      "text_snippet": "Revenue increased by 15% to $5.2M compared to..."
    }
  ],
  "confidence": 0.65,
  "message": null
}
```

**Response 200 — No relevant documents:**
```json
{
  "answer": null,
  "sources": [],
  "confidence": 0.0,
  "message": "No relevant documents found for your question."
}
```

**Response 200 — Below confidence threshold:**
```json
{
  "answer": null,
  "sources": [...],
  "confidence": 0.0,
  "message": "Retrieved documents did not meet the confidence threshold."
}
```

**Response 200 — LLM timeout:**
```json
{
  "answer": null,
  "sources": [...],
  "confidence": 0.65,
  "message": "Answer generation timed out. Sources are provided for reference."
}
```

**Error 422 — Invalid question:**
```json
{
  "detail": [{"msg": "String should have at least 1 character", ...}]
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the key findings?",
    "top_k": 5,
    "filter": {"source": "Q4 Financial Report"},
    "chat_history": [
      {"role": "user", "content": "Tell me about the Q4 report"},
      {"role": "assistant", "content": "The Q4 report covers..."}
    ]
  }'
```

---

## Global Error Responses

| Status | Condition | Response |
|---|---|---|
| 422 | Validation error (bad input) | `{"detail": "..."}` or Pydantic error array |
| 503 | Pinecone unavailable | `{"detail": "Vector store temporarily unavailable"}` |
| 504 | LLM generation timeout | `{"detail": "Answer generation timed out"}` |

---

## Environment Variables



| Variable | Default | Description |
|---|---|---|
| `RAG_PINECONE_API_KEY` | (required) | Pinecone API key |
| `RAG_PINECONE_INDEX_NAME` | `rag-index` | Pinecone index name |
| `RAG_GROQ_API_KEY` | (required) | Groq API key |
| `RAG_GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `RAG_EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_OPENAI_API_KEY` | (required) | OpenAI API key |
| `RAG_CONFIDENCE_THRESHOLD` | `0.3` | Min similarity score |
| `RAG_LLM_TIMEOUT` | `300` | LLM timeout (seconds) |
| `RAG_MAX_CHAT_HISTORY` | `1` | Max conversation messages in prompt |
| `RAG_MAX_FILE_SIZE` | `52428800` | Max upload size (bytes) |
| `RAG_DATABASE_URL` | `sqlite:///./rag_documents.db` | SQLite database path |

"""Document ingestion service orchestrating parsing, chunking, and vector storage."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.models.internal import DocumentRecord
from app.models.schemas import DocumentUploadRequest, DocumentUploadResponse
from app.services.document_store import DocumentStore
from app.services.file_parser import ParserFactory
from app.services.text_chunker import TextChunker
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    upload_request: DocumentUploadRequest,
    vector_store: VectorStoreService,
    document_store: DocumentStore,
    bm25_store=None,
) -> DocumentUploadResponse:
    """Ingest a document: parse, chunk, embed, and store in the vector database.

    Generates a unique document_id, persists the document record to SQLite,
    extracts text via the appropriate parser, splits into chunks, and upserts
    embeddings to Pinecone. BM25 chunks are saved to SQLite so the keyword
    index survives server restarts.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename.
        content_type: MIME type of the uploaded file.
        upload_request: Validated upload metadata (source, chunk_size, chunk_overlap).
        vector_store: VectorStoreService instance for embedding and storage.
        document_store: DocumentStore instance for SQLite persistence.
        bm25_store: Optional BM25Store instance for keyword-based retrieval.

    Returns:
        DocumentUploadResponse with document_id, filename, and chunk_count.

    Raises:
        ValueError: If the file type is unsupported or no text could be extracted.
    """
    document_id = str(uuid.uuid4())

    # Persist document record to SQLite before any vector ingestion
    record = DocumentRecord(
        id=document_id,
        filename=filename,
        content_type=content_type,
        source=upload_request.source,
        file_size=len(file_bytes),
        chunk_count=0,
        file_data=file_bytes,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
    await document_store.save(record)

    # Parse file to extract text
    parser = ParserFactory.get_parser(content_type)
    raw_text = await parser.parse(file_bytes, filename)

    if not raw_text or not raw_text.strip():
        raise ValueError("No text could be extracted from the uploaded file")

    # Chunk the extracted text
    chunker = TextChunker(
        chunk_size=upload_request.chunk_size,
        chunk_overlap=upload_request.chunk_overlap,
    )
    metadata = {"document_id": document_id, "source": upload_request.source}
    chunks = chunker.chunk(raw_text, metadata)

    # Embed and upsert to Pinecone; roll back on failure for atomicity
    try:
        stored_count = await vector_store.upsert_chunks(chunks, document_id)
    except Exception:
        try:
            await vector_store.delete_by_document_id(document_id)
        except Exception:
            logger.exception("Vector cleanup also failed for document %s", document_id)
        raise

    # ------------------------------------------------------------------
    # BM25: persist chunks to SQLite THEN update in-memory index
    # This order ensures the data is durable before we touch the index.
    # ------------------------------------------------------------------
    if bm25_store is not None:
        try:
            # 1. Persist to SQLite so chunks survive a restart
            await document_store.save_bm25_chunks(chunks)
            logger.info(
                "Saved %d BM25 chunks to SQLite for document %s",
                len(chunks),
                document_id,
            )

            # 2. Update the live in-memory index
            bm25_store.add_documents(chunks)
            logger.info(
                "BM25 in-memory index now has %d chunks total",
                bm25_store.chunk_count,
            )
        except Exception:
            logger.exception(
                "BM25 indexing failed for document %s — vector index is intact",
                document_id,
            )
            # Non-fatal: vector search still works without BM25

    # Update the persisted record with the actual chunk count
    await document_store.update_chunk_count(document_id, stored_count)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        chunk_count=stored_count,
        message="Document ingested successfully",
    )
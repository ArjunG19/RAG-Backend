"""FastAPI route definitions for the RAG backend API."""

from __future__ import annotations

import logging

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.models.schemas import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadataResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.document_store import DocumentStore
from app.services.ingestion_service import ingest_document
from app.services.rag_chain import RAGChain
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global cache for embeddings model to avoid reloading on every request
_embeddings_cache: dict[str, Any] = {}

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "image/png",
    "image/jpeg",
}


async def get_document_store() -> DocumentStore:
    """Dependency that provides a DocumentStore instance.

    Uses the database_url from application settings.
    Overridden in tests via dependency_overrides.
    """
    return DocumentStore(settings.database_url)


async def init_embeddings_cache() -> None:
    """Initialize the HuggingFace Inference API embeddings model cache at startup.
    
    Initializes the HuggingFace Inference API embeddings once and stores in cache
    to avoid reinitialization on every request.
    """
    if "embeddings" not in _embeddings_cache:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        logger.info("Initializing HuggingFace Inference API embeddings: %s", settings.embedding_model_name)
        embeddings_model = HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model_name,
            task="feature-extraction",
            huggingfacehub_api_token=settings.huggingface_api_key,
        )
        _embeddings_cache["embeddings"] = embeddings_model
        logger.info("HuggingFace Inference API embeddings initialized and cached")


async def get_vector_store() -> VectorStoreService:
    """Dependency that provides a VectorStoreService instance.

    Uses HuggingFace Inference API for cloud-based embeddings and Pinecone for storage.
    Overridden in tests via dependency_overrides.
    """
    from pinecone import Pinecone

    from app.services.embedding_service import EmbeddingService

    if "embeddings" not in _embeddings_cache:
        await init_embeddings_cache()
    
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    embedding_service = EmbeddingService(model=_embeddings_cache["embeddings"])
    return VectorStoreService(index=index, embedding_service=embedding_service)


async def get_rag_chain(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> RAGChain:
    """Dependency that provides a RAGChain instance.

    Uses Groq for LLM chat (cloud-based) and the injected VectorStoreService.
    Overridden in tests via dependency_overrides.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        timeout=settings.llm_timeout,
    )
    return RAGChain(llm=llm, vector_store=vector_store)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    vector_store: VectorStoreService = Depends(get_vector_store),
    document_store: DocumentStore = Depends(get_document_store),
) -> DocumentUploadResponse:
    """Upload a document for ingestion into the knowledge base.

    Accepts a multipart file upload with metadata form fields. Validates
    file content type and size, then delegates to the ingestion service.
    """
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {content_type}. Supported: pdf, text, png, jpeg",
        )

    # Read file bytes and validate size
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size:
        raise HTTPException(
            status_code=422,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes",
        )

    # Build the upload request from form fields
    try:
        upload_request = DocumentUploadRequest(
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Delegate to ingestion service
    try:
        return await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unknown",
            content_type=content_type,
            upload_request=upload_request,
            vector_store=vector_store,
            document_store=document_store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> QueryResponse:
    """Query the knowledge base using the RAG pipeline.

    Accepts a JSON QueryRequest body, runs the full RAG pipeline
    (retrieve → confidence check → generate → parse), and returns
    a structured QueryResponse.
    """
    return await rag_chain.query(
        question=request.question,
        top_k=request.top_k,
        filter=request.filter,
        chat_history=request.chat_history,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    document_store: DocumentStore = Depends(get_document_store),
) -> DocumentListResponse:
    """List all persisted documents with metadata."""
    records = await document_store.list_all()
    return DocumentListResponse(
        documents=[
            DocumentMetadataResponse(
                document_id=r.id,
                filename=r.filename,
                content_type=r.content_type,
                source=r.source,
                file_size=r.file_size,
                uploaded_at=r.uploaded_at,
            )
            for r in records
        ]
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> DocumentDetailResponse:
    """Retrieve metadata for a single document by ID."""
    record = await document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetailResponse(
        document_id=record.id,
        filename=record.filename,
        content_type=record.content_type,
        source=record.source,
        file_size=record.file_size,
        uploaded_at=record.uploaded_at,
        chunk_count=record.chunk_count,
    )


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
) -> Response:
    """Download the original file for a document."""
    record = await document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(
        content=record.file_data,
        media_type=record.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{record.filename}"',
        },
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    """Delete a document and its associated vectors."""
    record = await document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await document_store.delete(document_id)

    try:
        await vector_store.delete_by_document_id(document_id)
    except Exception:
        logger.exception(
            "Vector deletion failed for document %s", document_id
        )
        return {
            "message": "Document record deleted but vector cleanup failed",
            "document_id": document_id,
        }

    return {"message": "Document deleted successfully", "document_id": document_id}


@router.get("/health")
async def health_check() -> dict:
    """Return the operational status of the service."""
    return {"status": "healthy"}

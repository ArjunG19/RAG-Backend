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
from app.services.bm25_store import BM25Store
from app.services.document_store import DocumentStore
from app.services.ingestion_service import ingest_document
from app.services.rag_chain import RAGChain
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter()

# Singletons initialised at startup
_embeddings_cache: dict[str, Any] = {}
_bm25_cache: dict[str, BM25Store] = {}

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "image/png",
    "image/jpeg",
}


# ------------------------------------------------------------------
# Dependency providers
# ------------------------------------------------------------------

async def get_document_store() -> DocumentStore:
    """Dependency that provides a DocumentStore instance."""
    return DocumentStore(settings.database_url)


async def init_embeddings_cache() -> None:
    """Initialise the OpenAI embeddings model once and cache it."""
    if "embeddings" not in _embeddings_cache:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Initializing OpenAI embeddings: %s", settings.embedding_model_name)
        embeddings_model = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            openai_api_key=settings.openai_api_key,
        )
        _embeddings_cache["embeddings"] = embeddings_model
        logger.info("OpenAI embeddings initialized and cached")


async def init_bm25_cache() -> None:
    """Create the BM25Store singleton.

    The index is empty here; main.py calls rebuild_from_db after this.
    """
    if "bm25" not in _bm25_cache:
        _bm25_cache["bm25"] = BM25Store()
        logger.info("BM25Store singleton created")


async def get_bm25_store() -> BM25Store:
    """Dependency: return the singleton BM25Store."""
    if "bm25" not in _bm25_cache:
        _bm25_cache["bm25"] = BM25Store()
        logger.warning("BM25Store created on-demand (was not pre-initialised)")
    return _bm25_cache["bm25"]


async def get_vector_store() -> VectorStoreService:
    """Dependency: return a VectorStoreService backed by Pinecone."""
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
    bm25_store: BM25Store = Depends(get_bm25_store),
) -> RAGChain:
    """Dependency: return a RAGChain with hybrid retrieval."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        timeout=settings.llm_timeout,
    )
    return RAGChain(
        llm=llm,
        vector_store=vector_store,
        bm25_store=bm25_store,
        relevance_threshold=0.4,
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    vector_store: VectorStoreService = Depends(get_vector_store),
    document_store: DocumentStore = Depends(get_document_store),
    bm25_store: BM25Store = Depends(get_bm25_store),
) -> DocumentUploadResponse:
    """Upload a document and ingest it into both Pinecone and BM25 indexes."""
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {content_type}. Supported: pdf, text, png, jpeg",
        )

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size:
        raise HTTPException(
            status_code=422,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes",
        )

    try:
        upload_request = DocumentUploadRequest(
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        return await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unknown",
            content_type=content_type,
            upload_request=upload_request,
            vector_store=vector_store,
            document_store=document_store,
            bm25_store=bm25_store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> QueryResponse:
    """Query the knowledge base using the hybrid RAG pipeline."""
    logger.debug("query_documents received request with top_k=%s", request.top_k)
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
    bm25_store: BM25Store = Depends(get_bm25_store),
) -> dict:
    """Delete a document from SQLite, Pinecone, and the BM25 index."""
    record = await document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Delete document row (CASCADE deletes bm25_chunks rows too)
    await document_store.delete(document_id)

    # 2. Remove from live BM25 in-memory index
    removed = bm25_store.remove_document(document_id)
    logger.info("BM25Store: removed %d chunks for document %s", removed, document_id)

    # 3. Delete vectors from Pinecone
    try:
        await vector_store.delete_by_document_id(document_id)
    except Exception:
        logger.exception("Vector deletion failed for document %s", document_id)
        return {
            "message": "Document record and BM25 index deleted but vector cleanup failed",
            "document_id": document_id,
        }

    return {"message": "Document deleted successfully", "document_id": document_id}


@router.get("/health")
async def health_check() -> dict:
    """Return the operational status of the service."""
    bm25_store = _bm25_cache.get("bm25")
    return {
        "status": "healthy",
        "bm25_chunks": bm25_store.chunk_count if bm25_store else 0,
    }
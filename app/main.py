"""FastAPI application entry point with CORS, routes, and global error handlers."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import init_embeddings_cache, router
from app.config import settings
from app.services.document_store import DocumentStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend",
    description="LangChain-based Retrieval-Augmented Generation API",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire routes
app.include_router(router)


@app.on_event("startup")
async def startup_init() -> None:
    """Initialize resources on startup: database and HuggingFace embeddings cache."""
    # Initialize database
    store = DocumentStore(settings.database_url)
    await store.init_db()
    await store.close()
    
    # Initialize embeddings cache (HuggingFace Inference API)
    await init_embeddings_cache()


@app.exception_handler(asyncio.TimeoutError)
async def timeout_error_handler(request: Request, exc: asyncio.TimeoutError) -> JSONResponse:
    """Handle LLM timeout errors with HTTP 504."""
    logger.error("Request timed out: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=504,
        content={"detail": "Answer generation timed out"},
    )


@app.exception_handler(TimeoutError)
async def builtin_timeout_handler(request: Request, exc: TimeoutError) -> JSONResponse:
    """Handle built-in TimeoutError with HTTP 504."""
    logger.error("Request timed out: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=504,
        content={"detail": "Answer generation timed out"},
    )


try:
    from pinecone.exceptions import PineconeException

    @app.exception_handler(PineconeException)
    async def pinecone_error_handler(
        request: Request, exc: PineconeException
    ) -> JSONResponse:
        """Handle Pinecone unavailability with HTTP 503."""
        logger.error("Pinecone error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Vector store temporarily unavailable"},
        )
except ImportError:
    pass


try:
    from pinecone.core.openapi.shared.exceptions import ServiceException

    @app.exception_handler(ServiceException)
    async def pinecone_service_error_handler(
        request: Request, exc: ServiceException
    ) -> JSONResponse:
        """Handle Pinecone service errors with HTTP 503."""
        logger.error("Pinecone service error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Vector store temporarily unavailable"},
        )
except ImportError:
    pass


@app.exception_handler(ConnectionError)
async def connection_error_handler(
    request: Request, exc: ConnectionError
) -> JSONResponse:
    """Handle connection errors (e.g. Pinecone unreachable) with HTTP 503."""
    logger.error("Connection error: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Vector store temporarily unavailable"},
    )

# Mount frontend static files AFTER API routes so API takes priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

"""Vector store service for Pinecone vector storage and retrieval."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from pinecone import Pinecone  # type: ignore[attr-defined]

from app.models.internal import RetrievalResult, TextChunk
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]  # seconds


class VectorStoreService:
    """Interface with Pinecone for vector storage and retrieval.

    Handles batch embedding and upserting of document chunks, similarity
    search with optional metadata filtering, and deletion by document ID.
    Includes retry logic with exponential backoff for Pinecone failures.
    """

    def __init__(self, index, embedding_service: EmbeddingService) -> None:
        self.index = index
        self.embedding_service = embedding_service

    async def upsert_chunks(self, chunks: list[TextChunk], document_id: str) -> int:
        """Embed and store chunks in Pinecone.

        Generates embeddings for all chunks in batch, then upserts vectors
        with metadata (document_id, source, chunk_index) to Pinecone.
        Retries on Pinecone failures with exponential backoff.

        Args:
            chunks: List of text chunks to embed and store.
            document_id: Unique identifier for the parent document.

        Returns:
            Count of vectors successfully stored.

        Raises:
            Exception: If all retry attempts fail.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.embed_documents(texts)

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = f"{document_id}#{chunk.chunk_index}"
            metadata = {
                "document_id": document_id,
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            vectors.append((vector_id, embedding, metadata))

        await self._upsert_with_retry(vectors)
        return len(vectors)

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k similar chunks with scores and metadata.

        Embeds the query string, performs a similarity search against
        Pinecone, and returns results ordered by descending score.

        Args:
            query: The query text to search for.
            top_k: Maximum number of results to return.
            filter: Optional Pinecone metadata filter dict.

        Returns:
            List of RetrievalResult ordered by descending similarity score.

        Raises:
            Exception: If all retry attempts fail.
        """
        query_embedding = await self.embedding_service.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter is not None:
            query_kwargs["filter"] = filter

        response = await self._query_with_retry(**query_kwargs)

        results = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            results.append(
                RetrievalResult(
                    text=metadata.get("text", ""),
                    score=match.get("score", 0.0),
                    metadata={
                        "document_id": metadata.get("document_id", ""),
                        "source": metadata.get("source", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors associated with a document ID.

        Used for atomicity support during ingestion — if ingestion fails
        partway through, this method cleans up any partial vectors.

        Args:
            document_id: The document ID whose vectors should be deleted.

        Raises:
            Exception: If all retry attempts fail.
        """
        await self._delete_with_retry(
            filter={"document_id": {"$eq": document_id}}
        )

    async def _upsert_with_retry(self, vectors: list[tuple]) -> None:
        """Upsert vectors to Pinecone with exponential backoff retry.

        Args:
            vectors: List of (id, embedding, metadata) tuples.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        last_exception: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self.index.upsert(vectors=vectors)
                return
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Pinecone upsert failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exception  # type: ignore[misc]

    async def _query_with_retry(self, **kwargs: Any) -> dict:
        """Query Pinecone with exponential backoff retry.

        Args:
            **kwargs: Arguments passed to Pinecone index.query().

        Returns:
            Pinecone query response dict.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        last_exception: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.index.query(**kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Pinecone query failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exception  # type: ignore[misc]

    async def _delete_with_retry(self, **kwargs: Any) -> None:
        """Delete from Pinecone with exponential backoff retry.

        Args:
            **kwargs: Arguments passed to Pinecone index.delete().

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        last_exception: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self.index.delete(**kwargs)
                return
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Pinecone delete failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exception  # type: ignore[misc]

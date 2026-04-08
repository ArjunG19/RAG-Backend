from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from app.models.internal import RetrievalResult, TextChunk
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]

# Hybrid scoring constants
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
BM25_ONLY_CAP = 0.3
SIGMOID_K = 1.0
SIGMOID_MIDPOINT = 3.0


class VectorStoreService:
    def __init__(self, index, embedding_service: EmbeddingService) -> None:
        self.index = index
        self.embedding_service = embedding_service

    async def upsert_chunks(self, chunks: list[TextChunk], document_id: str) -> int:
        if not chunks:
            return 0

        total = 0
        BATCH_SIZE = 100

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [chunk.text for chunk in batch]

            embeddings = await self.embedding_service.embed_documents(texts)

            vectors = []
            for chunk, embedding in zip(batch, embeddings):
                vector_id = f"{document_id}#{chunk.chunk_index}"
                metadata = {
                    "document_id": document_id,
                    "source": chunk.metadata.get("source", ""),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
                vectors.append((vector_id, embedding, metadata))

            await self._upsert_with_retry(vectors)
            total += len(vectors)

        return total

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        bm25_store=None,
    ) -> list[RetrievalResult]:

        logger.debug(f"similarity_search called with top_k={top_k}")

        # ---------- Dense Retrieval (single Pinecone call) ----------
        # Fetch a wider candidate pool to maximize overlap with BM25 results.
        # The final top_k selection happens after hybrid scoring.
        candidate_limit = max(top_k * 4, 40)

        query_embedding = await self.embedding_service.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": candidate_limit,
            "include_metadata": True,
        }
        if filter:
            query_kwargs["filter"] = filter

        response = await self._query_with_retry(**query_kwargs)

        # ---------- BM25 Retrieval (single call) ----------
        bm25_raw_results = []
        if bm25_store:
            bm25_raw_results = bm25_store.search(query, top_k=candidate_limit)

        # ---------- Collect candidates ----------
        # Vector candidates (keep original scores)
        vector_candidates = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            vector_candidates.append({
                "result": RetrievalResult(
                    text=metadata.get("text", ""),
                    score=match.get("score", 0.0),
                    metadata={
                        "document_id": metadata.get("document_id", ""),
                        "source": metadata.get("source", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                ),
                "vec_raw_score": match.get("score", 0.0),
                "bm25_raw_score": 0.0,
            })

        # BM25 candidates (keep original scores)
        bm25_candidates = []
        for text, metadata, score in bm25_raw_results:
            bm25_candidates.append({
                "result": RetrievalResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                ),
                "vec_raw_score": 0.0,
                "bm25_raw_score": score,
            })

        logger.debug(f"BM25 raw scores: {[c['bm25_raw_score'] for c in bm25_candidates[:5]]}")

        # ---------- Merge all candidates ----------
        all_candidates = {}
        for candidate in vector_candidates + bm25_candidates:
            doc_id = candidate['result'].metadata.get('document_id', '')
            chunk_idx = int(candidate['result'].metadata.get('chunk_index', 0))
            result_id = f"{doc_id}#{chunk_idx}"
            if result_id not in all_candidates:
                all_candidates[result_id] = candidate
            else:
                existing = all_candidates[result_id]
                existing["vec_raw_score"] = max(existing["vec_raw_score"], candidate["vec_raw_score"])
                existing["bm25_raw_score"] = max(existing["bm25_raw_score"], candidate["bm25_raw_score"])

        # ---------- Normalize and score ----------
        def sigmoid_normalize(score: float) -> float:
            """Sigmoid normalization for BM25 scores. Returns 0.0 for zero scores."""
            if score == 0.0:
                return 0.0
            return 1.0 / (1.0 + math.exp(-SIGMOID_K * (score - SIGMOID_MIDPOINT)))

        for candidate in all_candidates.values():
            vec_score = candidate["vec_raw_score"]
            bm25_raw = candidate["bm25_raw_score"]
            bm25_norm = sigmoid_normalize(bm25_raw)

            # Conditional weighting
            if bm25_norm > 0:
                hybrid_score = VECTOR_WEIGHT * vec_score + BM25_WEIGHT * bm25_norm
            else:
                hybrid_score = vec_score

            # Cap BM25-only candidates
            if vec_score == 0.0:
                hybrid_score = min(hybrid_score, BM25_ONLY_CAP)

            candidate["hybrid_score"] = hybrid_score
            candidate["result"].score = hybrid_score
            candidate["result"].vec_score = vec_score
            candidate["result"].bm25_score = bm25_norm

        # ---------- Select top results ----------
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:top_k]

        final_results = [c["result"] for c in sorted_candidates]
        logger.debug(f"Collected {len(vector_candidates)} vector + {len(bm25_candidates)} BM25 candidates, merged to {len(all_candidates)}, returning top {len(final_results)}")

        return final_results

    async def delete_by_document_id(self, document_id: str) -> None:
        await self._delete_with_retry(
            filter={"document_id": {"$eq": document_id}}
        )

    async def _upsert_with_retry(self, vectors: list[tuple]) -> None:
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.to_thread(self.index.upsert, vectors=vectors)
                return
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Upsert failed (%d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        raise last_exception

    async def _query_with_retry(self, **kwargs: Any) -> dict:
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return self.index.query(**kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Query failed (%d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        raise last_exception

    async def _delete_with_retry(self, **kwargs: Any) -> None:
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
                        "Delete failed (%d/%d), retrying in %ds: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        raise last_exception
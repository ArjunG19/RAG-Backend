"""BM25 keyword search store with SQLite-backed persistence.

The index is rebuilt from SQLite on every startup, so it survives
server restarts without any separate file management.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """In-memory BM25 index backed by SQLite for persistence.

    Workflow
    --------
    Startup  → call ``await rebuild_from_db(document_store)``
    Ingest   → call ``add_documents(chunks)``   (in-memory only;
               the caller must also call ``document_store.save_bm25_chunks``)
    Query    → call ``search(query, top_k)``
    Delete   → call ``remove_document(document_id)``  (in-memory only;
               the caller must also call
               ``document_store.delete_bm25_chunks_by_document``)
    """

    def __init__(self) -> None:
        # Parallel lists — index i in each list belongs to the same chunk
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        logger.debug("BM25Store instance created (empty; call rebuild_from_db on startup)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, split, drop single-char tokens."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _rebuild_index(self) -> None:
        """Rebuild the BM25Okapi index from current tokenized docs."""
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
            logger.debug("BM25 index built with %d chunks", len(self._tokenized))
        else:
            self._bm25 = None
            logger.debug("BM25 index cleared (no chunks)")

    # ------------------------------------------------------------------
    # Startup: rebuild from DB
    # ------------------------------------------------------------------

    async def rebuild_from_db(self, document_store) -> None:
        """Load all persisted chunks from SQLite and rebuild the in-memory index.

        Call this once during application startup AFTER the DB is initialised.

        Args:
            document_store: A DocumentStore instance (already initialised).
        """
        rows = await document_store.load_all_bm25_chunks()

        self._texts.clear()
        self._metadatas.clear()
        self._tokenized.clear()

        for row in rows:
            tokens = self._tokenize(row["text"])
            self._texts.append(row["text"])
            self._metadatas.append(
                {
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "source": row["source"],
                }
            )
            self._tokenized.append(tokens)

        self._rebuild_index()
        logger.info(
            "BM25Store rebuilt from DB: %d chunks across documents",
            len(self._texts),
        )

    # ------------------------------------------------------------------
    # Ingest path (in-memory only — caller persists to DB separately)
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list) -> None:
        """Add TextChunk objects to the in-memory index.

        The caller is responsible for also calling
        ``document_store.save_bm25_chunks(chunks)`` so the data
        survives a restart.

        Args:
            chunks: List of TextChunk with .text, .chunk_index,
                    and .metadata dict.
        """
        if not chunks:
            return

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._texts.append(chunk.text)
            self._tokenized.append(tokens)
            self._metadatas.append(
                {
                    "document_id": chunk.metadata.get("document_id", ""),
                    "source": chunk.metadata.get("source", ""),
                    "chunk_index": chunk.chunk_index,
                }
            )

        self._rebuild_index()
        logger.info(
            "BM25Store: added %d chunks (total %d)", len(chunks), len(self._texts)
        )

    # ------------------------------------------------------------------
    # Delete path (in-memory only — caller removes from DB separately)
    # ------------------------------------------------------------------

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document from the in-memory index.

        The caller is responsible for also calling
        ``document_store.delete_bm25_chunks_by_document(document_id)``.

        Args:
            document_id: The document whose chunks should be removed.

        Returns:
            Number of chunks removed.
        """
        keep_indices = [
            i
            for i, m in enumerate(self._metadatas)
            if m.get("document_id") != document_id
        ]
        removed = len(self._texts) - len(keep_indices)

        if removed == 0:
            logger.debug("BM25Store.remove_document: no chunks found for %s", document_id)
            return 0

        self._texts = [self._texts[i] for i in keep_indices]
        self._metadatas = [self._metadatas[i] for i in keep_indices]
        self._tokenized = [self._tokenized[i] for i in keep_indices]

        self._rebuild_index()
        logger.info(
            "BM25Store: removed %d chunks for document %s (total now %d)",
            removed,
            document_id,
            len(self._texts),
        )
        return removed

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """Return the top-k BM25 matches for a query.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (text, metadata, score) tuples sorted by score descending.
            Returns an empty list when the index is empty or query has no tokens.
        """
        if self._bm25 is None or not self._texts:
            logger.debug("BM25 search: index empty, returning no results")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            logger.debug("BM25 search: query produced no tokens")
            return []

        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(self._texts, self._metadatas, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        results = ranked[:top_k]
        logger.debug(
            "BM25 search: query=%r tokens=%s top_score=%.4f results=%d",
            query,
            tokens[:5],
            results[0][2] if results else 0.0,
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Total number of chunks currently in the index."""
        return len(self._texts)

    def __repr__(self) -> str:
        return f"BM25Store(chunks={self.chunk_count}, indexed={self._bm25 is not None})"
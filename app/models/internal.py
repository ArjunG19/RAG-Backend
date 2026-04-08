"""Internal data models used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """A segment of parsed document text with associated metadata.

    Attributes:
        text: The chunk text content.
        metadata: Dictionary containing document_id, source, and chunk_index.
        chunk_index: Positional index of this chunk within the parent document.
    """

    text: str
    metadata: dict  # {document_id, source, chunk_index}
    chunk_index: int


@dataclass
class RetrievalResult:
    """A single result from a similarity search against the vector store.

    Attributes:
        text: The retrieved chunk text.
        score: Combined hybrid score using conditional weighting:
            if bm25_norm > 0: score = 0.7 * vec_score + 0.3 * bm25_norm
            if bm25_norm == 0: score = vec_score (no penalty for absent BM25).
            BM25-only candidates (vec_score == 0) are capped at 0.3.
        metadata: Dictionary containing document_id, source, and chunk_index.
        vec_score: Individual vector/embedding similarity score (0-1).
        bm25_score: Sigmoid-normalized BM25 keyword match score (0-1).
    """

    text: str
    score: float
    metadata: dict
    vec_score: float = 0.0
    bm25_score: float = 0.0


@dataclass
class DocumentRecord:
    """A persisted document record stored in SQLite.

    Attributes:
        id: UUID string uniquely identifying the document.
        filename: Original upload filename.
        content_type: MIME type of the file (e.g. application/pdf).
        source: User-provided source label.
        file_size: Size of the raw file in bytes.
        chunk_count: Number of chunks stored in the vector store.
        file_data: Raw file bytes.
        uploaded_at: ISO 8601 timestamp of when the document was uploaded.
    """

    id: str
    filename: str
    content_type: str
    source: str
    file_size: int
    chunk_count: int
    file_data: bytes
    uploaded_at: str

"""Text chunking service using LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models.internal import TextChunk


class TextChunker:
    """Split raw text into overlapping token-based chunks.

    Chunk size and overlap are measured in tokens (via tiktoken) rather than
    characters, so limits align with the embedding model's context window and
    BM25 term frequencies stay consistent across chunks of varying word length.

    Each chunk carries propagated source metadata and an ascending chunk_index.
    """

    def __init__(
        self,
        chunk_size: int = settings.default_chunk_size,
        chunk_overlap: int = settings.default_chunk_overlap,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=settings.embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str, metadata: dict) -> list[TextChunk]:
        """Split *text* into chunks, each carrying source *metadata*.

        Returns a list of :class:`TextChunk` ordered by ascending
        ``chunk_index``.  Every chunk's metadata dict contains the
        caller-supplied keys **plus** ``chunk_index``.
        """
        raw_chunks: list[str] = self.splitter.split_text(text)

        return [
            TextChunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": idx},
                chunk_index=idx,
            )
            for idx, chunk_text in enumerate(raw_chunks)
        ]
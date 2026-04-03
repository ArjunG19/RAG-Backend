"""Embedding service for generating vector embeddings from text."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class EmbeddingService:
    """Wraps a LangChain Embeddings model to provide batch and single-query embedding.

    Accepts any LangChain-compatible Embeddings instance, allowing the caller
    to swap providers (OpenAI, HuggingFace, etc.) without changing service code.
    """

    def __init__(self, model: Embeddings) -> None:
        self.model = model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks.

        Args:
            texts: List of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        return await self.model.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text to embed.

        Returns:
            A single embedding vector.
        """
        return await self.model.aembed_query(text)

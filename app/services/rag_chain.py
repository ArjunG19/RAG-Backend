"""RAG chain service for orchestrating retrieval-augmented generation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.models.schemas import QueryResponse, Source
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information to answer, say so clearly.

--- CONTEXT ---
{context}
--- END CONTEXT ---

{history_section}Question: {question}

Provide a clear, concise answer. Reference the source documents when possible."""


HISTORY_TEMPLATE = """--- CONVERSATION HISTORY ---
{history}
--- END HISTORY ---

"""


class RAGChain:
    """Orchestrates the RAG pipeline: retrieve, confidence check, generate, parse.

    Retrieves relevant context from the vector store, checks confidence,
    formats a prompt, generates an answer via LLM, and returns a structured
    QueryResponse with source references.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStoreService,
        confidence_threshold: float | None = None,
        llm_timeout: int | None = None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )
        self.llm_timeout = (
            llm_timeout if llm_timeout is not None else settings.llm_timeout
        )

    async def query(
        self,
        question: str,
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        chat_history: Optional[list] = None,
    ) -> QueryResponse:
        """Run the full RAG pipeline: retrieve → confidence check → generate → parse.

        Args:
            question: The user's question.
            top_k: Maximum number of retrieval results.
            filter: Optional Pinecone metadata filter.
            chat_history: Optional list of recent ChatMessage dicts for conversation context.

        Returns:
            A QueryResponse with answer, sources, confidence, and optional message.
        """
        # Step 1: Retrieve relevant context
        results = await self.vector_store.similarity_search(
            query=question,
            top_k=top_k,
            filter=filter,
        )

        # Step 2: Confidence gate — no results
        if not results:
            msg = "No relevant documents found for your question."
            return QueryResponse(
                answer=msg,
                sources=[],
                confidence=0.0,
                message=msg,
            )

        # Confidence is the max similarity score
        confidence = max(r.score for r in results)

        # Step 3: Confidence gate — all scores below threshold
        if confidence < self.confidence_threshold:
            msg = "Retrieved documents did not meet the confidence threshold."
            return QueryResponse(
                answer=msg,
                sources=self._build_sources(results),
                confidence=0.0,
                message=msg,
            )

        # Step 4: Format prompt with context and optional chat history
        relevant_results = [r for r in results if r.score >= self.confidence_threshold]
        relevant_sorted = sorted(relevant_results, key=lambda r: r.score, reverse=True)
        context_parts = []
        for r in relevant_sorted:
            context_parts.append(
                f"[Source: {r.metadata.get('source', '')}, Score: {r.score:.2f}]\n{r.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build conversation history section
        history_section = ""
        if chat_history:
            max_history = settings.max_chat_history
            recent = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
            history_lines = []
            for msg in recent:
                role = msg.role if hasattr(msg, "role") else msg.get("role", "")
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                label = "User" if role == "user" else "Assistant"
                history_lines.append(f"{label}: {content}")
            if history_lines:
                history_section = HISTORY_TEMPLATE.format(history="\n".join(history_lines))

        formatted_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            history_section=history_section,
        )

        # Step 5: Generate answer via LLM (with timeout)
        try:
            answer = await asyncio.wait_for(
                self.llm.ainvoke(formatted_prompt),
                timeout=self.llm_timeout,
            )
            answer_text = answer.content if hasattr(answer, "content") else str(answer)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("LLM generation timed out after %ds", self.llm_timeout)
            return QueryResponse(
                answer=None,
                sources=self._build_sources(results),
                confidence=confidence,
                message="Answer generation timed out. Sources are provided for reference.",
            )

        # Step 6: Build response
        return QueryResponse(
            answer=answer_text,
            sources=self._build_sources(results),
            confidence=confidence,
        )

    def _build_sources(self, results: list) -> list[Source]:
        """Build source references ordered by descending score.

        Args:
            results: List of RetrievalResult from the vector store.

        Returns:
            List of Source objects ordered by descending score.
        """
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return [
            Source(
                document_id=r.metadata.get("document_id", ""),
                source=r.metadata.get("source", ""),
                chunk_index=r.metadata.get("chunk_index", 0),
                score=r.score,
                text_snippet=r.text[:200],
            )
            for r in sorted_results
        ]

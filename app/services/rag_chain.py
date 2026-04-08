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

FALLBACK_PROMPT_TEMPLATE = """You are a helpful assistant. The user asked a question, but no relevant documents were found in the knowledge base.

You may ONLY answer if the question is a general conceptual question that you can answer confidently from general knowledge (e.g., "what is AI?", "explain machine learning").

You MUST NOT:
- Fabricate specific facts, statistics, dates, or numbers
- Invent domain-specific data, proprietary information, or organizational details
- Guess at answers that require specific factual information

If the question requires specific factual information that you cannot answer confidently from general knowledge alone, respond with exactly: CANNOT_ANSWER

Question: {question}
"""


class RAGChain:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStoreService,
        bm25_store=None,
        confidence_threshold: float | None = None,
        llm_timeout: int | None = None,
        relevance_threshold: float = 0.4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )
        self.llm_timeout = (
            llm_timeout if llm_timeout is not None else settings.llm_timeout
        )
        self.relevance_threshold = relevance_threshold

    async def query(
        self,
        question: str,
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        chat_history: Optional[list] = None,
    ) -> QueryResponse:

        # 🔥 Hybrid retrieval
        results = await self.vector_store.similarity_search(
            query=question,
            top_k=top_k,
            filter=filter,
            bm25_store=self.bm25_store,
        )

        if not results:
            msg = "No relevant documents found."
            return QueryResponse(
                answer=msg,
                sources=[],
                confidence=0.0,
                message=msg,
            )

        # 🔥 Filter by relevance threshold before LLM
        relevant_results = [r for r in results if r.score >= self.relevance_threshold]

        if not relevant_results:
            # Post-retrieval fallback: call LLM with restricted prompt
            fallback_prompt = FALLBACK_PROMPT_TEMPLATE.format(question=question)
            try:
                fallback_answer = await asyncio.wait_for(
                    self.llm.ainvoke(fallback_prompt),
                    timeout=self.llm_timeout,
                )
                fallback_text = fallback_answer.content if hasattr(fallback_answer, "content") else str(fallback_answer)
            except (asyncio.TimeoutError, TimeoutError):
                return QueryResponse(
                    answer=None,
                    sources=[],
                    confidence=0.0,
                    message="LLM timed out",
                )

            if "CANNOT_ANSWER" in fallback_text:
                return QueryResponse(
                    answer="I'm sorry, I don't have enough information to answer that question.",
                    sources=[],
                    confidence=0.0,
                    message="The system could not find relevant documents and cannot answer this question from general knowledge.",
                )

            return QueryResponse(
                answer=fallback_text,
                sources=[],
                confidence=0.0,
                message="This answer is based on general knowledge and is not grounded in the knowledge base.",
            )

        # 🔥 Better confidence (avg of top 3 relevant results)
        top_scores = [r.score for r in relevant_results[:3]]
        confidence = sum(top_scores) / len(top_scores)

        # 🔥 Use only relevant results for context
        relevant_sorted = sorted(relevant_results, key=lambda r: r.score, reverse=True)

        context_parts = []
        for r in relevant_sorted:
            context_parts.append(
                f"[Source: {r.metadata.get('source', '')}, Score: {r.score:.2f}]\n{r.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Chat history
        history_section = ""
        if chat_history:
            max_history = settings.max_chat_history
            recent = chat_history[-max_history:]

            history_lines = []
            for msg in recent:
                if hasattr(msg, "role"):
                    role = msg.role
                    content = msg.content
                else:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                label = "User" if role == "user" else "Assistant"
                history_lines.append(f"{label}: {content}")

            history_section = HISTORY_TEMPLATE.format(
                history="\n".join(history_lines)
            )

        formatted_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            history_section=history_section,
        )

        try:
            answer = await asyncio.wait_for(
                self.llm.ainvoke(formatted_prompt),
                timeout=self.llm_timeout,
            )
            answer_text = answer.content if hasattr(answer, "content") else str(answer)

        except (asyncio.TimeoutError, TimeoutError):
            return QueryResponse(
                answer=None,
                sources=self._build_sources(results),
                confidence=confidence,
                message="LLM timed out",
            )

        return QueryResponse(
            answer=answer_text,
            sources=self._build_sources(results),
            confidence=confidence,
        )

    def _build_sources(self, results: list) -> list[Source]:
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        return [
            Source(
                document_id=r.metadata.get("document_id", ""),
                source=r.metadata.get("source", ""),
                chunk_index=r.metadata.get("chunk_index", 0),
                score=r.score,
                vec_score=r.vec_score,
                bm25_score=r.bm25_score,
                text_snippet=r.text[:200],
            )
            for r in sorted_results
        ]
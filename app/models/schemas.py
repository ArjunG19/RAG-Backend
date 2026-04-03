"""Pydantic request and response models for the RAG API."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class FileType(str, Enum):
    """Supported file types for document upload."""

    PDF = "pdf"
    TEXT = "text"
    IMAGE = "image"


class DocumentUploadRequest(BaseModel):
    """Metadata sent alongside the uploaded file."""

    source: str = Field(..., min_length=1, description="Origin or label for the document")
    file_type: Optional[FileType] = Field(None, description="Explicit file type override")
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    @model_validator(mode="after")
    def check_overlap_less_than_size(self) -> DocumentUploadRequest:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class DocumentUploadResponse(BaseModel):
    """Response returned after successful document upload."""

    document_id: str
    filename: str
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter: Optional[Dict] = Field(default=None, description="Pinecone metadata filter")
    chat_history: Optional[List["ChatMessage"]] = Field(
        default=None,
        description="Recent conversation history for context (last N messages)",
    )


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class Source(BaseModel):
    """A single source reference in a query response."""

    document_id: str
    source: str
    chunk_index: int
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    """Structured response from the RAG query pipeline."""

    answer: Optional[str]
    sources: List[Source]
    confidence: float
    message: Optional[str] = None


class DocumentMetadataResponse(BaseModel):
    """Metadata for a persisted document (used in list responses)."""

    document_id: str
    filename: str
    content_type: str
    source: str
    file_size: int
    uploaded_at: str


class DocumentDetailResponse(DocumentMetadataResponse):
    """Full document metadata including chunk count (used in detail responses)."""

    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response wrapping a list of document metadata entries."""

    documents: List[DocumentMetadataResponse]

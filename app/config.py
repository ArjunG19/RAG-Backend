from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API keys
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    huggingface_api_key: str = Field(default="", description="HuggingFace API key")

    # Pinecone configuration
    pinecone_index_name: str = Field(
        default="rag-index", description="Pinecone index name"
    )
    pinecone_environment: str = Field(
        default="us-east-1", description="Pinecone environment"
    )

    # Embedding model (HuggingFace Inference API)
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for embeddings (via Inference API)",
    )

    # Groq configuration
    groq_model: str = Field(
        default="mixtral-8x7b-32768",
        description="Groq model name for chat generation",
    )

    # Chunking defaults
    default_chunk_size: int = Field(
        default=1000, ge=100, le=4000, description="Default chunk size in characters"
    )
    default_chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Default chunk overlap in characters"
    )

    # Confidence threshold
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence score for answers"
    )

    # LLM timeout
    llm_timeout: int = Field(
        default=30, ge=1, description="LLM request timeout in seconds"
    )

    # Max chat history messages to include in prompt
    max_chat_history: int = Field(
        default=10, ge=0, le=50,
        description="Max number of recent chat messages to include in LLM prompt",
    )

    # Max file size (bytes) - default 50 MB
    max_file_size: int = Field(
        default=50 * 1024 * 1024, ge=1, description="Maximum upload file size in bytes"
    )

    # Database configuration
    database_url: str = Field(
        default="sqlite:///./rag_documents.db",
        description="SQLite database URL for document persistence",
    )

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "extra": "ignore"}


settings = Settings()

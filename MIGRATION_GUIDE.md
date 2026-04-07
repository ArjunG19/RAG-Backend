# Migration to Cloud-Based LLM & Embeddings

This guide covers the changes made to switch from local Ollama + local embeddings to **Groq LLM** + **HuggingFace Inference API** for full cloud-based deployment.

## What Changed

### 1. **LLM Provider**
- **Before**: Ollama (local, required running server)
- **After**: Groq (cloud-based, no local setup needed)

### 2. **Embedding Model**
- **Before**: SentenceTransformers (local model)
- **After**: HuggingFace Inference API (cloud-based)

### 3. **Vector Store**
- **Before**: Pinecone
- **After**: Pinecone (unchanged)

## Updated Dependencies

Removed:
- `langchain-ollama`
- `sentence-transformers`

Added:
- `langchain-groq>=0.1`

Updated:
- `langchain-huggingface>=0.1` (now uses HuggingFaceEndpointEmbeddings instead of HuggingFaceEmbeddings)

## Environment Variables

### Old Configuration (Ollama-based)
```env
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_OLLAMA_MODEL=phi3:mini
RAG_EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### New Configuration (Cloud-based)
```env
RAG_GROQ_API_KEY=your_groq_api_key_here
RAG_GROQ_MODEL=mixtral-8x7b-32768

RAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RAG_HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## Setup Instructions

### 1. Get API Keys

#### Groq API Key
1. Visit https://console.groq.com/login
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy it to your `.env` file as `RAG_GROQ_API_KEY`

#### HuggingFace API Key
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with `read` access
3. Copy it to your `.env` file as `RAG_HUGGINGFACE_API_KEY`

### 2. Update Environment Variables

Update your `.env` file with the new API keys:

```env
# Groq LLM (Cloud-based)
RAG_GROQ_API_KEY=your_groq_api_key_here
RAG_GROQ_MODEL=mixtral-8x7b-32768

# HuggingFace Inference API (Cloud-based embeddings)
RAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RAG_HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Keep your existing Pinecone credentials
RAG_PINECONE_API_KEY=your_pinecone_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m uvicorn app.main:app --reload
```

No Ollama server needed! Everything is cloud-hosted.

## Code Changes Summary

### [app/config.py](app/config.py)
- Removed: `ollama_base_url`, `ollama_model`
- Added: `groq_api_key`, `groq_model`, `huggingface_api_key`
- Updated: `embedding_model_name` default to use HuggingFace format

### [app/api/routes.py](app/api/routes.py)
- **`init_embeddings_cache()`**: Changed from `HuggingFaceEmbeddings` to `HuggingFaceEndpointEmbeddings` (Inference API)
- **`get_rag_chain()`**: Changed from `ChatOllama` to `ChatGroq`
- **`get_vector_store()`**: Updated docstring to reflect cloud-based embeddings

### [app/main.py](app/main.py)
- Updated startup comment to reflect HuggingFace Inference API instead of SentenceTransformer

### [requirements.txt](requirements.txt)
- Removed: `langchain-ollama>=0.2`, `sentence-transformers>=2.0`
- Added: `langchain-groq>=0.1`

## Available Groq Models

Groq offers several fast models. Update `RAG_GROQ_MODEL` as needed:
- `mixtral-8x7b-32768` (default, good balance)
- `llama-3.1-70b-versatile` (more capable, slower)
- `llama-3.1-8b-instant` (faster, smaller)
- `gemma-7b-it` (compact)

Check https://console.groq.com/docs/models for current availability.

## Available HuggingFace Embedding Models

Popular embedding models available via HuggingFace Inference API:
- `sentence-transformers/all-MiniLM-L6-v2` (default, recommended)
- `sentence-transformers/all-mpnet-base-v2` (better quality, slower)
- `sentence-transformers/distiluse-base-multilingual-cased-v2` (multilingual)

## Cost Considerations

- **Groq**: Free tier with generous limits, excellent for development/testing
- **HuggingFace Inference**: Free tier available for popular models
- **Pinecone**: Existing costs remain unchanged

## Deployment

This setup is **fully cloud-based** and can be deployed anywhere:
- Docker container
- AWS Lambda
- Heroku
- Google Cloud Functions
- Any serverless platform

No need to manage local Ollama servers anymore!

## Rollback (if needed)

If you need to revert to Ollama:
1. Run: `git checkout requirements.txt app/config.py app/api/routes.py app/main.py`
2. Update environment variables back to Ollama format
3. Install original dependencies: `pip install langchain-ollama sentence-transformers`
4. Restart the application

## Testing

Verify everything works:

```bash
# Health check
curl http://localhost:8000/health

# Check API documentation
open http://localhost:8000/docs
```

Upload a document and try the `/query` endpoint. The system should work exactly as before, but now with cloud-based services!

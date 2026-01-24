# Tech Stack Overview

## 🎯 Project Purpose
RAG (Retrieval-Augmented Generation) Question Generation Engine for CBSE curriculum - generates exam-quality questions from PDF documents.

---

## 🔧 Core Technologies

### **Web Framework & API**
- **FastAPI** (v0.115.6)
  - REST API framework
  - Async request handling
  - Automatic OpenAPI documentation
  - Request validation with Pydantic

- **Uvicorn** (v0.30.6)
  - ASGI server for running FastAPI
  - Production-ready HTTP server

### **Database**
- **PostgreSQL** (via psycopg3)
  - Primary relational database
  - Stores: Corpora, Documents, Chunks, Jobs, Generated Questions
  - Connection pooling with SQLModel

- **SQLModel** (v0.0.22)
  - ORM built on SQLAlchemy + Pydantic
  - Type-safe database models
  - Database migrations with Alembic

### **Vector Database & Similarity Search**
- **FAISS** (Facebook AI Similarity Search) - CPU version (v1.9.0.post1)
  - Vector similarity search for RAG retrieval
  - Index type: `IndexFlatIP` (Inner Product for cosine similarity)
  - Stores embeddings on disk per corpus/class/subject
  - Fast approximate nearest neighbor search

### **Embedding Model**
- **Sentence Transformers** (v3.3.1)
  - Library for generating text embeddings
  - **Model**: `sentence-transformers/all-MiniLM-L6-v2` (default)
    - 384-dimensional embeddings
    - Optimized for semantic similarity
    - Normalized embeddings for cosine similarity

### **LLM (Large Language Model) Providers**
Supports multiple providers via unified interface:

1. **OpenRouter** (Primary)
   - Unified API for multiple LLM providers
   - Default model: `openai/gpt-4o-mini`
   - Supports: OpenAI, Anthropic, Google, and more

2. **Google Gemini** (Alternative)
   - Model: `gemini-2.5-flash`
   - Direct API integration

3. **Stub LLM** (Testing)
   - Mock LLM for development/testing

### **PDF Processing**
- **pdfplumber** (v0.11.5)
  - Primary PDF text extraction
  - Page-by-page text extraction
  - Preserves text structure

- **PyPDF2** (v3.0.1)
  - Additional PDF processing capabilities

### **Data Processing**
- **NumPy** (v1.22.4+)
  - Vector operations
  - Embedding array handling
  - FAISS integration

### **Configuration & Settings**
- **Pydantic Settings** (v2.7.0)
  - Environment variable management
  - Type-safe configuration
  - `.env` file support

### **HTTP Client**
- **httpx** (v0.28.1)
  - Async HTTP client for LLM API calls
  - Retry logic with Tenacity

### **Retry Logic**
- **Tenacity** (v9.0.0)
  - Retry decorators for API calls
  - Handles transient failures

---

## 📊 Observability & Monitoring

### **Logging**
- **Structlog** (v24.4.0)
  - Structured JSON logging
  - Context-aware logging
  - Production-ready logging

### **Metrics**
- **Prometheus Client** (v0.21.1)
  - HTTP request metrics (counters, histograms)
  - Latency tracking
  - `/metrics` endpoint for Prometheus scraping

### **Rate Limiting**
- **SlowAPI** (v0.1.9)
  - Rate limiting middleware
  - Configurable requests per minute
  - Default: 120 requests/minute

---

## 🔐 Security & Authentication

- **API Key Middleware**
  - Optional API key authentication
  - Configurable via environment variables

- **CORS Middleware**
  - Cross-origin resource sharing
  - Configurable allowed origins

---

## 🗂️ Storage

### **Local File Storage** (Current)
- File system storage for PDFs
- Local index storage (FAISS indices)
- Configurable base paths:
  - `PDF_BASE_PATH`: PDF storage
  - `INDEX_BASE_PATH`: FAISS index storage
  - `UPLOADS_BASE_PATH`: Uploaded files

---

## 🧪 Development & Testing

### **Code Quality**
- **Ruff** (v0.8.4)
  - Fast Python linter
  - Code formatting

- **mypy** (v1.14.1)
  - Static type checking
  - Type safety validation

### **Testing**
- **pytest** (v8.3.4)
  - Test framework
- **pytest-asyncio** (v0.25.0)
  - Async test support

### **Database Migrations**
- **Alembic** (v1.13.3)
  - Database schema migrations
  - Version control for database changes

---

## 📦 Data Serialization

- **orjson** (v3.10.12)
  - Fast JSON serialization
  - Used for API responses

- **JSON Schema** (v4.23.0)
  - Schema validation
  - Request/response validation

---

## 🚀 Deployment

- **Docker Compose**
  - Containerized deployment
  - Separate API and Worker services
  - Environment variable configuration

---

## 📋 Technology Summary by Function

| Function | Technology |
|----------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **Database** | PostgreSQL + SQLModel |
| **Vector Search** | FAISS (CPU) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM** | OpenRouter (OpenAI GPT-4o-mini) / Gemini |
| **PDF Parsing** | pdfplumber + PyPDF2 |
| **Logging** | Structlog |
| **Metrics** | Prometheus |
| **Rate Limiting** | SlowAPI |
| **HTTP Client** | httpx |
| **Retry Logic** | Tenacity |
| **Type Checking** | mypy |
| **Linting** | Ruff |
| **Testing** | pytest |

---

## 🔄 Data Flow

1. **PDF Upload** → pdfplumber extracts text
2. **Chunking** → Syllabus-aware chunking creates text chunks
3. **Embedding** → Sentence Transformers generates vector embeddings
4. **Indexing** → FAISS creates similarity search index
5. **Storage** → PostgreSQL stores metadata, FAISS stores vectors
6. **Retrieval** → Query → Embed → FAISS search → Retrieve chunks
7. **Generation** → LLM (OpenRouter/Gemini) generates questions from retrieved context

---

## 📝 Notes

- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Similarity Metric**: Cosine similarity (via Inner Product)
- **Index Type**: Flat Index (exact search, suitable for moderate dataset sizes)
- **Default LLM**: OpenAI GPT-4o-mini via OpenRouter
- **Storage Backend**: Local file system (can be extended to S3/cloud storage)

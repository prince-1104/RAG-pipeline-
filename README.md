
# Niyukti RAG Question Generation Engine

RAG (Retrieval-Augmented Generation) question generation API for CBSE PDFs.

### 1) Setup environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment
Copy `.env.example` to `.env` in the project root directory and update with your values:
```bash
# LLM Provider (see .env.example for valid modes)
LLM_PROVIDER=openai_fallback_gemini

# OpenAI (primary)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Gemini (optional fallback)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Database
DATABASE_URL=postgresql://user:password@host:port/database?sslmode=require

# Redis
REDIS_URL=redis://localhost:6379

# Storage (choose one)
# Local:
# STORAGE_BACKEND=local
# PDF_BASE_PATH=data/pdfs
# UPLOADS_BASE_PATH=.data/uploads
# R2:
STORAGE_BACKEND=r2
R2_ACCOUNT_ID=your_r2_account_id
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
R2_BUCKET_NAME=your_bucket_name
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_REGION=auto

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=niyukti_knowledge

# Embeddings
EMBEDDING_API_KEY=your_openai_api_key_here
```

**Important:** Never commit the `.env` file to git. It contains sensitive credentials.

### 3) Prepare CBSE PDFs directory structure
Create the following structure under your repo root:
```
data/
  pdfs/
    cbse/
      8/
        Hindi/
          hindi_textbook_8.pdf
        English/
          english_textbook_8.pdf
        Social Science/
          social_science_textbook_8.pdf
        Mathematics/
          mathematics_textbook_8.pdf
        Science/
          science_textbook_8.pdf
      9/
        Hindi/
          hindi_textbook_9.pdf
        English/
          english_textbook_9.pdf
        Social Science/
          social_science_textbook_9.pdf
        Mathematics/
          mathematics_textbook_9.pdf
        Science/
          science_textbook_9.pdf
      10/
        Hindi/
          hindi_textbook_10.pdf
        English/
          english_textbook_10.pdf
        Social Science/
          social_science_textbook_10.pdf
        Mathematics/
          mathematics_textbook_10.pdf
        Science/
          science_textbook_10.pdf
```

**Important**: PDFs must be placed directly in `data/pdfs/cbse/{class}/{subject}/{filename}`. The service reads from these paths and does NOT copy them.

**Note**: Subject names with spaces (like "Social Science") are supported. Use the exact subject name when registering documents via API.

### 4) Initialize database
```bash
# Create tables
python -c "from app.db.session import init_db; init_db()"
```

### 5) Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 6) Run worker (async indexing/generation jobs)
```bash
python -m app.worker
```

### 7) Register a corpus and document
```bash
# Create a corpus
curl -X POST http://localhost:8080/api/v1/corpora \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test-tenant",
    "name": "CBSE Class 9-10 Corpus",
    "description": "Science and Mathematics textbooks"
  }'

# Register a PDF (replace CORPUS_ID and adjust filename)
# Example: Register Class 9 Science textbook
curl -X POST http://localhost:8080/api/v1/corpora/{CORPUS_ID}/documents/register \
  -H "Content-Type: application/json" \
  -d '{
    "class_level": 9,
    "subject": "Science",
    "filename": "science_textbook_9.pdf"
  }'

# Example: Register Class 9 Social Science textbook (note: subject name with space)
curl -X POST http://localhost:8080/api/v1/corpora/{CORPUS_ID}/documents/register \
  -H "Content-Type: application/json" \
  -d '{
    "class_level": 9,
    "subject": "Social Science",
    "filename": "social_science_textbook_9.pdf"
  }'
```

### 8) Index documents
```bash
# Create indexing job
curl -X POST http://localhost:8080/api/v1/corpora/{CORPUS_ID}/index-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test-tenant"
  }'

# Check job status
curl http://localhost:8080/api/v1/jobs/{JOB_ID}
```

### 9) Generate questions (pools)
```bash
curl -X POST http://localhost:8080/v1/question-generation/pools \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test-tenant",
    "corpus_id": "{CORPUS_ID}",
    "blueprint": {
      "class_level": 9,
      "subject": "Science",
      "chapters": ["Chapter 1: Matter in Our Surroundings"],
      "difficulty_mix": {"easy": 2, "medium": 3, "hard": 1},
      "question_types": ["mcq", "short_answer"],
      "count": 6,
      "language": "en"
    }
  }'
```

### 10) Open API docs
- `http://localhost:8080/docs`

**Note:** `/rag/*` endpoints are internal/admin-only and require API key auth.

## Running in Local PDF Mode

### Directory Layout
**Important**: `data/` folder is at the **repo root** (same level as `app/`), not inside `app/`.

```
niyukti-que-gen-Engine/
├── app/                  # Your code
├── data/                 # ← At repo root
│   ├── pdfs/             # Source CBSE PDFs (read-only)
│   │   └── cbse/
│   │       ├── 8/
│   │       │   ├── Hindi/
│   │       │   ├── English/
│   │       │   ├── Social Science/
│   │       │   ├── Mathematics/
│   │       │   └── Science/
│   │       ├── 9/
│   │       │   ├── Hindi/
│   │       │   ├── English/
│   │       │   ├── Social Science/
│   │       │   ├── Mathematics/
│   │       │   └── Science/
│   │       └── 10/
│   │           ├── Hindi/
│   │           ├── English/
│   │           ├── Social Science/
│   │           ├── Mathematics/
│   │           └── Science/
├── .data/
│   └── uploads/          # Future uploads (if needed)
├── config/
├── docs/
└── ...
```

### Key Points
- **PDFs are NOT uploaded**: They must exist at `data/pdfs/cbse/{class}/{subject}/{filename}` before registration
- **PDFs are read-only**: The service never modifies source PDFs
- **Qdrant is required**: All embeddings are stored in Qdrant (local or managed)
- **Path traversal protection**: The service rejects any PDF paths outside `PDF_BASE_PATH`

### Configuration
Set these in `.env`:
- `STORAGE_BACKEND=local` (required for local mode)
- `PDF_BASE_PATH=data/pdfs` (where CBSE PDFs live)
- `QDRANT_URL=http://localhost:6333` (local Qdrant)
- `QDRANT_COLLECTION=niyukti_knowledge`
- `EMBEDDING_API_KEY=...` (OpenAI embeddings)
- `DATABASE_URL=...` (Postgres connection string)

### Failure Modes
- **PDF not found**: Check that the file exists at `data/pdfs/cbse/{class}/{subject}/{filename}`
- **Invalid PDF**: File must start with `%PDF` header
- **Path traversal**: Filename cannot contain `/`, `\`, or start with `.`
- **No chunks in Qdrant**: Run indexing job first before generating questions

## Configuration
Use `.env.example` as a template. Create `.env` in the project root directory. Key variables:
- `DATABASE_URL` - Postgres connection string
- `REDIS_URL` - Redis connection string
- `STORAGE_BACKEND=local|r2` - Storage backend
- `PDF_BASE_PATH` - Local PDF path (when using `local`)
- `QDRANT_URL` - Qdrant connection string
- `QDRANT_COLLECTION` - Qdrant collection name
- `EMBEDDING_API_KEY` - OpenAI embeddings key for indexing/retrieval
- `LLM_PROVIDER` - LLM provider mode (see `.env.example`)
- `OPENAI_API_KEY` - Required for OpenAI usage
- `GEMINI_API_KEY` - Required for Gemini usage

## Documentation
Internal documentation is maintained outside this repository.



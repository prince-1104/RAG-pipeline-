
### 1) Setup environment


python -m venv .venv
.\.venv\Scripts\activate  

pip install -r requirements.txt
```
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload


python -m app.worker



### 2) Configure environment
Copy `.env.example` to `.env` in the project root directory and update with your values:
```bash
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Required: PostgreSQL Database URL
DATABASE_URL=postgresql://user:password@host:port/database?sslmode=require

# Optional: Other settings (defaults provided)
LLM_PROVIDER=gemini
STORAGE_BACKEND=local
PDF_BASE_PATH=data/pdfs
INDEX_BASE_PATH=data/indexes
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

### 9) Generate questions
```bash
curl -X POST http://localhost:8080/api/v1/generation/sync \
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
│   └── indexes/          # FAISS indices (auto-generated)
│       └── {corpus_id}/
│           └── class_{N}/
│               └── {subject}/
│                   ├── index.faiss
│                   ├── meta.json
│                   └── vecs.npy
├── .data/
│   └── uploads/          # Future uploads (if needed)
├── config/
├── docs/
└── ...
```

### Key Points
- **PDFs are NOT uploaded**: They must exist at `data/pdfs/cbse/{class}/{subject}/{filename}` before registration
- **PDFs are read-only**: The service never modifies source PDFs
- **Indices are persisted**: FAISS indices are saved to `data/indexes/` and survive restarts
- **Path traversal protection**: The service rejects any PDF paths outside `PDF_BASE_PATH`

### Configuration
Set these in `.env`:
- `STORAGE_BACKEND=local` (required for local mode)
- `PDF_BASE_PATH=data/pdfs` (where CBSE PDFs live)
- `INDEX_BASE_PATH=data/indexes` (where FAISS indices are stored)
- `DATABASE_URL=...` (Postgres connection string)

### Failure Modes
- **PDF not found**: Check that the file exists at `data/pdfs/cbse/{class}/{subject}/{filename}`
- **Invalid PDF**: File must start with `%PDF` header
- **Path traversal**: Filename cannot contain `/`, `\`, or start with `.`
- **Index not found**: Run indexing job first before generating questions

## Configuration
Use `config/env.example` as a template. Create `.env` in the project root directory. Key variables:
- `DATABASE_URL` - Postgres connection string
- `STORAGE_BACKEND=local` - Use local PDF mode (no S3)
- `PDF_BASE_PATH` - Where CBSE PDFs are stored
- `INDEX_BASE_PATH` - Where FAISS indices are persisted
- `LLM_PROVIDER=stub|gemini` - LLM provider (stub for testing)
- `GEMINI_API_KEY` - Required if `LLM_PROVIDER=gemini`

## Key docs
- **[Comprehensive Documentation](docs/COMPREHENSIVE_DOCUMENTATION.md)** - Complete feature documentation and roadmap
- **[Features Summary](docs/FEATURES_SUMMARY.md)** - Quick reference of implemented and planned features
- `docs/architecture.md` - System architecture details
- `docs/api-contract.md` - API endpoint specifications
- `docs/data-model.md` - Database schema documentation
- `docs/rag-design.md` - RAG pipeline design
- `docs/question-blueprint.md` - Question blueprint structure
- `docs/prompting-gemini.md` - Prompt engineering details
- `docs/guardrails.md` - Quality control mechanisms
- `docs/ops.md` - Operations and deployment guide
- `POSTMAN_TESTING_GUIDE.md` - API testing guide



# Admin Question Generation Feature

## Overview

This feature allows admins to trigger question generation for uploaded PDFs with custom criteria. This is the **ONLY** place where LLM calls are allowed for question generation.

## Key Constraints

- ✅ **Admin-triggered generation is the ONLY place LLM calls are allowed**
- ❌ **PDF upload must NEVER trigger LLM**
- ❌ **Institute APIs must NEVER enqueue generation jobs**
- ✅ **Admin endpoints are asynchronous and return immediately**
- ✅ **Question generation logic is reused, not duplicated**

## Architecture

### Flow

1. **Admin uploads PDF** → `/api/v1/ingest` → PDF stored to R2 → Indexing job queued (chunking, embedding, FAISS indexing) → Response includes `corpus_id`
2. **Admin triggers question generation** → `/api/v1/admin/generate-questions` → Job queued → Worker processes → LLM generates questions → Questions stored in DB + Redis
3. **Institute requests questions** → `/api/v1/generate-questions` with `X-Corpus-ID` → Checks Redis → Falls back to DB → Returns questions

### Components

- **Admin Endpoint**: `app/api/v1/routes/admin.py`
  - Endpoint: `POST /api/v1/admin/generate-questions`
  - Validates request and creates async job
  - Returns immediately with job_id

- **Task Handler**: `app/tasks/admin_generate_job.py`
  - Processes admin generation jobs
  - Generates questions for each class in the specified range
  - Stores questions in PoolQuestion table
  - Adds questions to Redis pools

- **Worker**: `app/worker.py`
  - Handles `JobKind.admin_generate_questions` jobs
  - Calls task handler which performs LLM generation

## API Usage

### Request

```bash
POST /api/v1/admin/generate-questions
Content-Type: application/json
X-Tenant-ID: niyukti-platform

{
  "document_id": "a41a313c1992...",
  "from_class": 9,
  "to_class": 10,
  "subject": "Science",
  "difficulty": "medium",
  "question_count": 100,
  "question_types": ["mcq", "short"],
  "chapters": null  // Optional: null means use all chapters
}
```

### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Question generation job queued for document a41a313c1992.... Generating 100 medium questions for classes 9-10 in Science."
}
```

### Pool Status (Read-only)

```bash
GET /api/v1/pool/status?class_level=10&subject=Science&syllabus=CBSE&syllabus_version=2024
X-Corpus-ID: <corpus-id>
```

Example response:
```json
{
  "corpus_id": "8287ed29-cbf9-4696-989e-c0d190035499",
  "class_level": 10,
  "subject": "Science",
  "status": "generating",
  "pools": {
    "easy": { "status": "generating", "count": 0 },
    "medium": { "status": "generating", "count": 0 },
    "hard": { "status": "ready", "count": 14 }
  }
}
```

Optional: filter to a single chapter:
```bash
GET /api/v1/pool/status?class_level=10&subject=Science&chapter=Chapter%201&syllabus=CBSE&syllabus_version=2024
X-Corpus-ID: <corpus-id>
```

### Check Job Status

```bash
GET /api/v1/jobs/{job_id}
```

Response includes:
- `status`: queued | running | succeeded | failed
- `result_json`: Contains total questions generated, pools updated, etc.
  - Ingestion responses include `corpus_id` and must be persisted by callers.

## Request Schema

### AdminGenerateQuestionsRequest

- `document_id` (required): Document ID for which to generate questions
- `from_class` (required): Starting class level (1-12)
- `to_class` (required): Ending class level (1-12, inclusive)
- `subject` (required): Subject name (e.g., "Science", "Mathematics")
- `difficulty` (required): "easy" | "medium" | "hard"
- `question_count` (required): Number of questions to generate (1-500)
- `question_types` (optional): List of question types, default: `["mcq", "short"]`
  - Options: "mcq", "short", "long"
- `chapters` (optional): Specific chapters to focus on
  - If `null`, auto-detects chapters from chunks
  - If provided, uses only those chapters

## How It Works

1. **Admin calls endpoint** with criteria
2. **Endpoint validates**:
   - Document exists
   - Corpus exists and tenant matches
   - Class range is valid
   - Document has required metadata

3. **Job created** with `JobKind.admin_generate_questions`

4. **Worker processes job**:
   - For each class in range (from_class to to_class):
     - Auto-detects chapters (if not provided)
     - Calculates questions per class (distributed evenly)
     - Calls `generate_questions_sync()` (reuses existing logic)
     - Stores questions in PoolQuestion table
     - Adds question IDs to Redis pool

5. **Questions available**:
   - Stored in database (PoolQuestion table)
   - Cached in Redis for fast retrieval
   - Institutes can fetch via existing `/api/v1/generate-questions` endpoint (requires `X-Corpus-ID`)

## Database Schema

### New JobKind

Added `admin_generate_questions` to `JobKind` enum:

```python
class JobKind(str, enum.Enum):
    ingest = "ingest"
    index = "index"
    generate = "generate"
    generate_question_pool = "generate_question_pool"
    refresh_question_pool = "refresh_question_pool"
    admin_generate_questions = "admin_generate_questions"  # NEW
```

### Storage

Questions are stored in:
- **PoolQuestion** table: Full question data with metadata
- **Redis**: Question IDs cached for fast retrieval
- **QuestionPool**: Pool metadata (syllabus, class, subject, difficulty)

## Benefits

1. **Pre-generation**: Questions generated upfront, making institute requests faster
2. **Flexible**: Admin can specify class range, difficulty, question types
3. **Reusable**: Uses existing generation logic (`generate_questions_sync`)
4. **Scalable**: Async processing, doesn't block admin requests
5. **Cached**: Questions stored in Redis for fast retrieval

## Example Workflow

1. Admin uploads PDF for Class 9-10 Science
2. PDF is indexed (chunked, embedded, FAISS indexed)
3. Admin triggers question generation:
   ```bash
   POST /api/v1/admin/generate-questions
   {
     "document_id": "...",
     "from_class": 9,
     "to_class": 10,
     "subject": "Science",
     "difficulty": "medium",
     "question_count": 200
   }
   ```
4. Job runs in background, generates 200 questions (100 per class)
5. Questions stored in DB and Redis
6. Institute requests questions (with `X-Corpus-ID`) → Fast retrieval from Redis

## Error Handling

- **Document not found**: Returns 404
- **Corpus not found**: Returns 404
- **Tenant mismatch**: Returns 403
- **Invalid class range**: Returns 400
- **Missing metadata**: Returns 400
- **Generation failures**: Logged, job marked as failed, other classes continue

## Notes

- Questions are distributed evenly across classes in the range
- If chapters are not specified, they are auto-detected from chunks
- Questions reuse existing generation pipeline (same quality, guardrails, validation)
- Redis pools are keyed by: `qp:{syllabus}:{class}:{subject}:{difficulty}:{version}`
- Pools are chapter-scoped; each chapter has its own pool per difficulty
- Questions are shuffled before being added to Redis for randomization

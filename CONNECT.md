# RAG Question-Generation Service — Integration Guide

This document describes what this service does, how to connect to it from any client (including the Niyukti platform), which endpoints to use, and what metadata and headers are required so integrations remain stable.

---

## 1. What This Service Does

This is a **RAG (Retrieval-Augmented Generation) question-generation microservice** that:

| Capability | Description |
|------------|-------------|
| **PDF ingestion** | Accepts curriculum PDFs (multipart upload), stores them (e.g. Cloudflare R2), and runs an async pipeline: chunking → embedding → indexing into **Qdrant**. |
| **Question generation** | Generates assessment questions (MCQ and descriptive) from indexed content using LLMs (OpenAI, Gemini, OpenRouter). Generation can be **async** (job-based) or **sync** (runtime RAG). |
| **Question pools** | Pre-generates and caches questions by syllabus, class, subject, chapter, and difficulty. Clients can **serve from pools** for fast responses without LLM calls. |
| **Runtime RAG** | On-demand question generation: query vector store → retrieve chunks → call LLM → return structured questions. |
| **Multi-tenant** | Supports tenants via `X-Tenant-ID`. Niyukti uses `niyukti-platform`; other projects can use their own tenant IDs. |
| **Corpus management** | Organizes documents and questions by **corpus** (e.g. one per tenant/board). Ingestion can auto-create a default corpus per tenant. |

**Stack:** FastAPI, PostgreSQL (metadata + jobs + pools), Qdrant (vectors), Redis (pool cache), R2/local storage (PDFs), configurable LLM providers.

---

## 2. Base URL and API Layout

Assume the service is deployed at a base URL, e.g. `https://rag-service.example.com`. All paths below are relative to that base.

| Prefix | Purpose |
|--------|---------|
| `/health`, `/healthz` | Liveness/readiness (no auth). |
| `/api/v1/*` | **Niyukti-compatible API** — same contract Niyukti uses. Use this if you want to integrate like Niyukti. |
| `/v1/*` | **Generic v1 API** — corpora, jobs, question-generation jobs, ingest, admin, school. |
| `/rag/*` | **Runtime RAG API** — synchronous generate-questions from vector store + LLM. |

OpenAPI (Swagger) is at `/openapi.json`; ReDoc at `/redoc`.

---

## 3. Authentication and Headers

### 3.1 API key (optional)

If the server is configured with `API_KEY` or `API_KEYS` (comma-separated) in the environment:

- **Required** for all requests under `/v1/`, `/api/v1/`, and `/rag/` (except paths listed below). 
- Send the key in one of:
  - **Header:** `X-API-Key: <your-api-key>`
  - **Header:** `Authorization: Bearer <your-api-key>`

**Paths that skip API key check:**  
`/health`, `/healthz`, `/v1/health`, `/v1/healthz`, `/api/v1/health`, `/api/v1/healthz`, `/docs`, `/openapi.json`, `/redoc`, `/favicon.ico`.

### 3.2 Tenant and corpus identification

| Header | Required | Description |
|--------|----------|-------------|
| `X-Tenant-ID` | For school endpoints and when using Niyukti-style flows | Tenant identifier. Niyukti uses `niyukti-platform`. Defaults to `niyukti-platform` on Niyukti ingest/generate if omitted. |
| `X-Corpus-ID` | For pool status and generate-questions when not using board+chapter lookup | Corpus UUID. If omitted in `/api/v1/generate-questions`, the service may resolve corpus from `board` + `class_level` + `subject` (and optionally `chapter`). |
| `X-After-Serial-No` | Optional | Integer; only return questions with `serial_no` greater than this (pagination / deduplication). |
| `X-Internal-Request` | For `/v1/question-generation/pools` | Must be `true` for pool serving (internal use). |

**Important:** For a stable integration, prefer sending **`X-Corpus-ID`** when you know it (e.g. after ingest returns `corpus_id`). That avoids ambiguity when multiple corpora match the same board/class/subject/chapter.

---

## 4. Endpoints Reference

### 4.1 Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` or `/healthz` | No | Returns `{"status": "ok", "service": "...", "env": "...", ...}`. |

Use for liveness/readiness probes.

---

### 4.2 Niyukti-compatible API (`/api/v1`)

These are the endpoints the **Niyukti platform** uses. Any project that wants the same behavior and response shapes should use these.

#### Ingest PDF

| Method | Path | Auth | Content-Type |
|--------|------|------|--------------|
| POST | `/api/v1/ingest` or `/api/v1/ingest/pdf` | Yes* | `multipart/form-data` |

**Form fields (required):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | PDF file. |
| `doc_id` | string | Yes | Your document ID (unique per document). |
| `board` | string | Yes | Board identifier (e.g. `CBSE`). |
| `class_level` | int | Yes | Class 1–12. |
| `subject` | string | Yes | Subject name (e.g. `science`, `maths`). |

**Form fields (optional but recommended for correct chapter scoping):**

| Field | Type | Description |
|-------|------|-------------|
| `chapter` | string | Chapter name. |
| `chapter_name` | string | Chapter title/name. |
| `chapter_number` | int | Chapter number. |

**Headers:**  
`X-Tenant-ID` (optional; defaults to `niyukti-platform`).

**Response (202):**

```json
{
  "success": true,
  "doc_id": "<doc_id>",
  "corpus_id": "<uuid>",
  "storage_uri": "...",
  "job_id": "<uuid>",
  "index_job_id": "<uuid>",
  "document_id": "<uuid>"
}
```

**Important:** Store `corpus_id` and `index_job_id`. Poll job status with **GET /v1/jobs/{index_job_id}** (see §4.3). Indexing must complete before generate-questions can use the new content.

---

#### Ingest stats (Niyukti UI)

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/v1/ingest/stats` | Yes* |

**Headers:** `X-Tenant-ID` (optional).

**Response:**  
`tenant_id`, `corpus_id`, `pdfs_uploaded`, `chunks_created`, `last_index_job` (job_id, status, timestamps, progress).

---

#### Generate questions (Niyukti)

| Method | Path | Auth | Content-Type |
|--------|------|------|--------------|
| POST | `/api/v1/generate-questions` | Yes* | `application/json` |

**Request body (NiyuktiGenerateRequest):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class_level` | int | Yes | 1–12. |
| `subject` | string | Yes | e.g. `maths`, `science`, `social_science`. |
| `board` | string | No* | Required if `X-Corpus-ID` is not sent. e.g. `CBSE`. |
| `chapter` | string or int | No* | Required if `X-Corpus-ID` is not sent and you need chapter scoping. |
| `mcq_count` | int | No** | Number of MCQs (default 0). |
| `descriptive_count` | int | No** | Number of descriptive questions (default 0). |
| `question_count` | int | No | Total questions; used with `mcq_ratio` if counts not provided. |
| `mcq_ratio` | float | No | 0–100; MCQ share when using `question_count`. |
| `difficulty_level` | string | No | `easy` \| `medium` \| `hard` when not using per-difficulty counts. |
| `easy_count` | int | No | Number of easy questions. |
| `medium_count` | int | No | Number of medium questions. |
| `hard_count` | int | No | Number of hard questions. |

\* Either send **`X-Corpus-ID`** (recommended) or **`board`** (and optionally `chapter`) so the service can resolve the corpus.  
\** Either (`mcq_count` + `descriptive_count`) or (`easy_count` + `medium_count` + `hard_count`) or `question_count` + `mcq_ratio` must yield a positive number of questions.

**Headers:**  
`X-Tenant-ID` (optional, default `niyukti-platform`), `X-Corpus-ID` (recommended), `X-After-Serial-No` (optional).

**Response (200):** List of questions in Niyukti shape:

```json
{
  "questions": [
    {
      "question_text": "...",
      "question_type": "mcq",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "class_level": 7,
      "subject": "science"
    },
    {
      "question_text": "...",
      "question_type": "descriptive",
      "correct_answer": "...",
      "expected_answer_text": "...",
      "expected_keywords": ["..."],
      "class_level": 7,
      "subject": "science"
    }
  ]
}
```

**Response (202):** Generation in progress:

```json
{
  "status": "generating",
  "message": "questions_are_being_generated_please_check_back_shortly",
  "corpus_id": "<uuid>",
  "class_level": 7,
  "subject": "science"
}
```

Client should retry the same request later when receiving 202.

**Subject mapping:** The service normalizes subject names (e.g. `maths` → `Mathematics`, `science` → `Science`, `sst` → `Social Science`). Sending the same subject names Niyukti uses ensures correct pool/chapter resolution.

---

#### Pool status (read-only)

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/v1/pool/status` | Yes* |

**Query parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `class_level` | int | Yes | 1–12. |
| `subject` | string | Yes | Subject name. |
| `syllabus` | string | No | Default `CBSE`. |
| `syllabus_version` | string | No | Default `2024`. |
| `chapter` | string | No | Optional chapter filter. |

**Headers:**  
`X-Corpus-ID` **required**. `X-Tenant-ID` optional.

**Response:**  
`corpus_id`, `syllabus`, `syllabus_version`, `class_level`, `subject`, `status` (e.g. `ready`, `ready_partial`, `generating`, `pending`, `failed`), and `pools` (per-difficulty status and counts).

---

#### Test endpoints (Niyukti)

- `GET /api/v1/test-simple` — returns `{"status": "ok"}`
- `POST /api/v1/test-post` — echo
- `GET /api/v1/test-error-handling` — forces 500 (for testing error handling)

---

### 4.3 Job status (index and generation)

Used to poll ingest/index and generation jobs. This is the **only** job-status endpoint (mounted under v1, not under `/api/v1`).

| Method | Path | Auth |
|--------|------|------|
| GET | `/v1/jobs/{job_id}` | Yes* |

**Response:**

```json
{
  "job_id": "<uuid>",
  "kind": "ingest|index|generate|...",
  "status": "queued|claimed|running|completed|completed_partial|failed_retryable|failed_terminal",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "error": { "code": "...", "message": "...", "retryable": true/false },
  "result": { ... }
}
```

For ingest, `result` includes `index_job_id`; poll that `job_id` to know when indexing is done.

---

### 4.4 Generic v1 API (corpora, ingest, generation jobs)

#### Corpora

- **POST /v1/corpora** — Create corpus (body: `name`, `description`, `tenant_id`). Returns `corpus_id`, `name`, `board`, `tenant_id`.
- **GET /v1/corpora/{corpus_id}** — Get corpus.

#### Ingest (generic)

- **POST /v1/ingest/pdf** — Same contract as Niyukti ingest but with optional form fields: `syllabus_version`, `language`, `source_type`, `tenant_id`. Returns 202 with `corpus_id`, `index_job_id`, etc.

#### Question generation (async job)

- **POST /v1/question-generation/jobs** — Enqueue a generation job (body: `GenerateRequest` with `tenant_id`, `corpus_id`, `blueprint`). Returns `job_id`, `status`.
- **GET /v1/question-generation/jobs/{job_id}/result** — Get result when status is `completed` or `completed_partial` (returns 409 until then).

Sync generation **POST /v1/question-generation** is disabled (403); use pools or RAG instead.

#### Pools (internal)

- **POST /v1/question-generation/pools** — Serve from pools. Requires `X-Internal-Request: true` and body `PoolRequest` (tenant_id, corpus_id, class_level, subject, difficulty_mix, syllabus, syllabus_version). Used internally; Niyukti uses `/api/v1/generate-questions` which may serve from pools under the hood.

---

### 4.5 Runtime RAG (sync generate)

| Method | Path | Auth | Content-Type |
|--------|------|------|--------------|
| POST | `/rag/generate/questions` | Yes* | `application/json` |

**Request body (RAGGenerateRequest):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `board` | string | Yes | e.g. `CBSE`. |
| `class` | int | Yes | 1–12 (JSON key is `class`). |
| `subject` | string | Yes | Subject name. |
| `chapter_number` | int | Yes | Chapter number. |
| `difficulty` | string | Yes | `easy` \| `medium` \| `hard`. |
| `question_type` | string | Yes | `mcq` \| `descriptive`. |
| `count` | int | Yes | 1–50. |

**Response:**  
`questions`: array of `{ stem, options?, answer?, qtype, difficulty }`.

Uses Qdrant + LLM at request time; no job polling.

---

### 4.6 School self-service (multitenant)

- **POST /v1/school/generate-questions** — Tenant-scoped generation; requires **X-Tenant-ID**. Body: `SchoolGenerateRequest` (corpus_id, subject, chapter, difficulty_mix, etc.). Returns questions in school schema. Use when you want tenant isolation and the school-specific prompt/schema.

---

## 5. How Niyukti Connects (Summary)

1. **Base URL:** Niyukti calls the same service base URL (e.g. `https://<rag-service>/api/v1`).
2. **Auth:** If the server sets `API_KEY`/`API_KEYS`, Niyukti sends `X-API-Key` or `Authorization: Bearer <key>`.
3. **Ingest:**  
   - **POST /api/v1/ingest** (or **/api/v1/ingest/pdf**) with `multipart/form-data`: `file`, `doc_id`, `board`, `class_level`, `subject`, and optionally `chapter`, `chapter_name`, `chapter_number`.  
   - Saves `corpus_id` and `index_job_id` from the response; polls **GET /v1/jobs/{index_job_id}** until indexing is complete.
4. **Generate questions:**  
   - **POST /api/v1/generate-questions** with JSON: `class_level`, `subject`, and either `X-Corpus-ID` or `board` (and optionally `chapter`). Optionally `mcq_count`/`descriptive_count` or `easy_count`/`medium_count`/`hard_count` (and `mcq_ratio`).  
   - On **200**, uses the returned `questions` array.  
   - On **202**, retries the same request later.
5. **Pool status:**  
   - **GET /api/v1/pool/status?class_level=...&subject=...** with **X-Corpus-ID** to show readiness per difficulty.
6. **Headers:** Niyukti sends **X-Tenant-ID** (e.g. `niyukti-platform`) where needed; for generate-questions it’s best to send **X-Corpus-ID** once known from ingest.

---

## 6. Metadata and Fields to Avoid Breakage

### 6.1 Ingest

- **Always send:** `doc_id`, `board`, `class_level`, `subject`.
- **Strongly recommended:** `chapter`, `chapter_name`, `chapter_number` so questions can be scoped by chapter and pool status is correct.
- If the server runs with **strict ingest metadata** (`INGEST_ENFORCE_STRICT_METADATA=true`), additional fields (e.g. `syllabus_version`, `language`, `source_type`) may be required; otherwise the service may reject with 400 and `missing_required_ingest_metadata`.

### 6.2 Generate-questions (Niyukti)

- **Corpus resolution:** Prefer **X-Corpus-ID** from ingest response. If not sent, provide **board** and optionally **chapter** so the service can resolve a single corpus (otherwise you may get 404 or 409).
- **Subject:** Use consistent names (e.g. `science`, `maths`, `social_science`); they are normalized internally.
- **Counts:** Ensure at least one of: (`mcq_count` + `descriptive_count`), or (`easy_count` + `medium_count` + `hard_count`), or `question_count` (with `mcq_ratio`) so that total requested count &gt; 0.

### 6.3 Pool status

- **X-Corpus-ID** is required. Same `class_level` and `subject` (and optional `chapter`) as used in generate-questions.

### 6.4 Runtime RAG

- All request fields are required; `class` must be 1–12, `chapter_number` ≥ 1, `count` 1–50. Use `class` in JSON (not `class_level`).

---

## 7. Error Handling

| Status | Meaning |
|--------|--------|
| 400 | Bad request: missing/invalid body or query (e.g. missing_corpus_metadata, missing_corpus_id, file_must_be_pdf). |
| 401 | Unauthorized: missing or invalid API key. |
| 403 | Forbidden: tenant_mismatch, pool_serving_internal_only, or sync_generation_disabled. |
| 404 | Not found: corpus_not_found, job_not_found, no_corpus_found_for_metadata. |
| 409 | Conflict: job_not_complete, multiple_corpus_matches. |
| 422 | Validation error: Pydantic/request validation (body/query/form). |
| 429 | Rate limited (e.g. RATE_LIMITED by slowapi). |
| 502 | Upstream/LLM or RAG pipeline error. |
| 500 | Server error; response may include `detail` and `error_type`. |

Always handle **202** on generate-questions as “in progress” and retry; treat **200** as success with a non-empty `questions` array when applicable.

---

## 8. Checklist for Connecting a New Project (Niyukti-like)

1. **Environment:** Configure service with storage (R2 or local), PostgreSQL, Qdrant, Redis, and LLM keys. Optionally set `API_KEY` or `API_KEYS` and use the same key from your app.
2. **Health:** Call `GET /health` or `GET /healthz` to verify service is up.
3. **Ingest:**  
   - POST to `/api/v1/ingest` (or `/api/v1/ingest/pdf`) with required form fields and recommended chapter metadata.  
   - Store `corpus_id` and `index_job_id`.  
   - Poll `GET /v1/jobs/{index_job_id}` (or `/api/v1/jobs/{index_job_id}` if mounted) until `status` is `completed` or `failed_terminal`.
4. **Generate questions:**  
   - POST to `/api/v1/generate-questions` with `class_level`, `subject`, and either `X-Corpus-ID` or `board` (+ optional `chapter`).  
   - Send `X-Tenant-ID` if you use tenants.  
   - On 200, use `response.questions`. On 202, retry after a delay.
5. **Optional:** Use `GET /api/v1/pool/status` with `X-Corpus-ID` to show pool readiness before calling generate-questions.
6. **Stability:** Use the same subject/board/class/chapter values as at ingest; prefer passing `X-Corpus-ID` once you have it to avoid resolution errors.

For a **non-Niyukti** integration you can instead use the generic **/v1** corpora/ingest/jobs and **/rag/generate/questions** with the request/response shapes documented above; the same auth and tenant/corpus rules apply where relevant.

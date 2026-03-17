## Overview

This service is a **FastAPI-based microservice** for:
- **Ingesting curriculum PDFs**, storing them in Cloudflare R2, and indexing them into **Qdrant** via an asynchronous pipeline.
- **Generating assessment questions** (via LLMs) into persistent question pools.
- **Serving questions** quickly from pre-generated pools (DB/Redis) or via a runtime RAG pipeline.

Core dependencies:
- **FastAPI** app in `app/main.py`.
- **PostgreSQL** (via `sqlmodel` / SQLAlchemy) for metadata, jobs, and question pools.
- **Qdrant** as the vector store.
- **Redis** for caching question pools.
- **Cloudflare R2** as object storage.
- **LLMs** (OpenAI, Gemini, OpenRouter) configured via `.env` / `app/settings.py`.

The service is designed to be called either:
- Directly by the **Niyukti platform** (compatibility endpoints under `/api/v1`), or
- As a **generic question-generation microservice** (v1 API under `/v1`, and RAG API under `/rag`).

---

## High-level Architecture

- **API Layer**
  - `app/main.py` creates the FastAPI app, configures middleware, and mounts routers.
  - `app/api/v1/router.py` is the **versioned API router** that aggregates the core v1 routes under `/v1`.
  - Additional top-level routers:
    - `/metrics` (metrics router, not shown here) for observability.
    - `/rag` for runtime RAG-based question generation.
    - `/api/v1` for **Niyukti compatibility** endpoints.

- **Domain / Modules**
  - `app/api/v1/routes/*.py` – HTTP endpoints for health, ingest, corpora, generation, jobs, admin, RAG, and Niyukti compatibility.
  - `app/db/models.py`, `app/db/session.py`, `app/db/migrations.py` – SQLModel models and DB session handling.
  - `app/indexing/*` – PDF parsing, chunking, embedding, Qdrant indexing pipeline.
  - `app/vector_store/*` – Qdrant integration.
  - `app/llm/*` – LLM client factory and provider-specific clients.
  - `app/pools/*` – Question pool management (promotion, prewarm, rebuild, status).
  - `app/generation/*` – Question-generation schemas, prompts, guardrails, validation.
  - `app/tasks/*` & `app/worker.py` – Background workers that execute queued jobs in `GenerationJob`.
  - `app/cache/*` – Redis client and caching helpers.
  - `app/storage/object_store.py` – Abstraction over local storage / R2.

- **Cross-cutting Concerns**
  - `app/settings.py` – Environment-driven configuration.
  - `app/observability/*` – Logging, metrics, rate limiting, AgentBill observability.
  - `app/api/v1/middleware/api_key_auth.py` – Optional API key authentication middleware.

---

## App Initialization & Runtime Behavior

Defined in `app/main.py`:

- **`create_app()`**
  - Loads configuration via `get_settings()`.
  - Configures logging and rate limiting (`slowapi`).
  - Creates FastAPI app with JSON as the default response.
  - Adds:
    - **CORS** middleware (currently `allow_origins=["*"]`).
    - **SlowAPIMiddleware** (`slowapi`) for rate limiting.
    - **APIKeyMiddleware** which enforces API key auth if `API_KEY` / `API_KEYS` are set.
  - Registers exception handlers for:
    - `RateLimitExceeded` (429),
    - `HTTPException`,
    - `RequestValidationError` (422),
    - generic `Exception` (500) with logging.
  - Adds an HTTP middleware that:
    - Measures request latency.
    - Records metrics via `observe_request`.

- **Startup (`@app.on_event("startup")`)**
  - Calls `init_db()` to initialize DB (logs warning if this fails, but doesn’t crash service).
  - Calls `ensure_qdrant_payload_indexes()` to ensure Qdrant index is set up.
  - Logs startup metadata (env, LLM provider).

- **Routers Mounted**
  - `app.include_router(metrics_router, include_in_schema=False)`
  - `app.include_router(v1_router, prefix="/v1")`
  - `app.include_router(health_router, tags=["health"])` → top-level `/healthz`, `/health`.
  - `app.include_router(rag_router, prefix="/rag", tags=["rag"])`
  - `app.include_router(niyukti_compat_router, prefix="/api/v1", tags=["niyukti-compat"])`

An ASGI server (e.g. `uvicorn`) typically runs `app.main:app`, listening on `api_host/api_port` from `Settings`.

---

## Configuration & Environment

Configuration is defined in `app/settings.py` as a `Settings` Pydantic model and loaded from `.env`. Important fields:

- **Service / Server**
  - `service_name`, `env`, `log_level`.
  - `api_host`, `api_port`.

- **Storage**
  - `storage_backend`: `"local"` or `"r2"`.
  - Local paths: `pdf_base_path`, `uploads_base_path`.
  - R2 config: `r2_account_id`, `r2_access_key_id`, `r2_secret_access_key`, `r2_bucket_name`, `r2_endpoint_url`, `r2_region`.

- **Vector DB & Cache**
  - `qdrant_url`, `qdrant_api_key`, `qdrant_collection`, `qdrant_timeout_s`.
  - `redis_url`, `redis_password`, `redis_socket_timeout`, `redis_socket_connect_timeout`.

- **LLM / Embeddings**
  - `embedding_provider`, `embedding_model`, `embedding_dim`, etc.
  - `llm_provider` (e.g. `stub`, `openai`, `chain`), `llm_primary_provider`, `llm_fallback_provider`, timeouts.
  - OpenAI / Gemini / OpenRouter API keys and models.
  - Generation tuning knobs: `generation_max_total_attempts`, `generation_max_batches`, guardrails toggles.

- **API / Auth**
  - `requests_per_minute` (rate limiting).
  - `api_key` and `api_keys` (comma-separated) for API key auth.

---

## Core Data Model & Job System (Conceptual)

The job system revolves around `GenerationJob` records in the DB:

- **Key enums (from `app/db/models.py` in simplified form)**
  - `JobKind`: `generate`, `index`, `ingest`, `admin_generate_questions`, etc.
  - `JobStatus`: `queued`, `running`, `completed`, `completed_partial`, `failed_retryable`, `failed_terminal`.

- **Main entities**
  - `Corpus`: logical grouping of documents (per tenant / board).
  - `Document`: metadata for each uploaded/registered PDF.
  - `Chunk`: text chunks created during indexing.
  - `GeneratedQuestion`: individual questions produced by LLMs.
  - `QuestionPool`, `PoolQuestion`: pre-generated pools of questions partitioned by syllabus, class, subject, chapter, difficulty, etc.

The workers (`app/worker.py`, `app/tasks/*`) poll `GenerationJob` records and:
- For **index jobs**:
  - Fetch PDFs from storage, parse into chunks.
  - Compute embeddings.
  - Upsert into Qdrant.
  - Update job status and `result_json`.
- For **generation jobs**:
  - Retrieve relevant chunks.
  - Call LLM using `app/llm/factory.py`.
  - Validate / normalize outputs (`app/generation/validation.py`).
  - Persist into `GeneratedQuestion` and/or `PoolQuestion` tables.

---

## API Surface – v1 Namespace (`/v1`)

Mounted via `app/api/v1/router.py`:

- **`/v1/health`**
  - **GET `/v1/healthz`**, **GET `/v1/health`**
  - Returns service status and basic, non-secret runtime info (LLM provider, models, AgentBill status).

- **`/v1/corpora` (`app/api/v1/routes/corpora.py`)**
  - **POST `/v1/corpora`** – Create a corpus.
    - Body: `CorpusCreateRequest`.
    - Writes `Corpus` record.
  - **GET `/v1/corpora/{corpus_id}`** – Get corpus metadata.
  - **POST `/v1/corpora/{corpus_id}/documents`**
    - Upload a PDF to object storage (R2) and create a `Document` record.
    - Only allowed when `storage_backend != "local"`.
  - **POST `/v1/corpora/{corpus_id}/documents/register`**
    - Register a **local** PDF (no upload); for `storage_backend == "local"`.
  - **POST `/v1/corpora/{corpus_id}/index-jobs`**
    - Enqueue a **Qdrant indexing job** (`JobKind.index`) over selected documents.
    - Returns `IndexJobResponse` with `job_id`.

- **`/v1/jobs` (`app/api/v1/routes/jobs.py`)**
  - **GET `/v1/jobs/{job_id}`**
    - Fetch status of any `GenerationJob` by ID (indexing, generation, ingest, etc.).
    - Returns `JobStatusResponse` with `status`, timestamps, error details, and `result`.

- **`/v1/question-generation` (`app/api/v1/routes/generation.py`)**
  - **POST `/v1/question-generation`**
    - Deprecated synchronous LLM generation; currently **disabled** (403).
  - **POST `/v1/question-generation/jobs`**
    - Enqueue an **asynchronous generation job** (`JobKind.generate`) based on a `GenerateRequest` blueprint.
    - Returns `GenerateJobResponse` (`job_id` + status).
  - **GET `/v1/question-generation/jobs/{job_id}/result`**
    - Fetch final `GenerateResponse` when the generation job is `completed`/`completed_partial`.
  - **POST `/v1/question-generation/pools`**
    - Internal-only endpoint to **serve questions from Redis-backed pools**.
    - Requires header `X-Internal-Request: true`.
    - Validates corpus/tenant; pulls from per-difficulty pools via `app.pools.service`.
    - Returns:
      - `status="ready"`/`"ready_partial"` + questions, or
      - `status="generating"` (202-like behavior) when pools are not ready.

- **`/v1/ingest` (`app/api/v1/routes/ingest.py`)**
  - **POST `/v1/ingest/pdf`**
    - Asynchronous **PDF ingestion** into R2 and indexing pipeline.
    - Validates metadata contract, ensures backend is `r2`, validates PDF magic header.
    - Writes:
      - `GenerationJob` of kind `ingest` (tracks entire ingest).
      - `Document` record with curriculum metadata.
      - A follow-up `GenerationJob` of kind `index`.
    - Returns `PDFIngestResponse`:
      - `success`, `doc_id`, `corpus_id`, `storage_uri`, `job_id` (ingest job), `index_job_id`, `document_id`.

- **`/v1/admin` (`app/api/v1/routes/admin.py`)**
  - Intended for **internal/admin use** only – this is where LLM question generation is allowed.

  - **GET `/v1/admin/generated-questions`**
    - Filterable read of `GeneratedQuestion` rows across various dimensions (job, document, class, subject, chapter, difficulty, qtype, language).

  - **POST `/v1/admin/generate-questions`**
    - Enqueue a **single admin generation job** (`JobKind.admin_generate_questions`) for one document.
    - Body: `AdminGenerateQuestionsRequest` (document, class range, subject, difficulty, question_count, question_types, optional chapters).
    - Returns `AdminGenerateQuestionsResponse` with job metadata.

  - **POST `/v1/admin/generate-questions/batch`**
    - Enqueue **multiple admin generation jobs** across multiple documents.
    - Handles difficulty `"any"` and maps `"descriptive"` type to short/long.
    - Returns `AdminBatchGenerateQuestionsResponse` with per-document job status.

  - **POST `/v1/admin/prewarm-pools`**
    - Global **promotion-only** endpoint:
      - Takes already-generated questions and promotes them into pools via `promote_all_generated_questions_to_pools`.
      - Creates a `prewarm_run` record and tracks counts of promoted questions and pools created.

  - **GET `/v1/admin/prewarm/{prewarm_id}/questions`**
    - List questions promoted in a specific prewarm run.

  - **POST `/v1/admin/invalidate-pools`**
    - Invalidate question pools for a syllabus/syllabus_version (e.g. when curriculum changes).

  - **POST `/v1/admin/reindex`**
    - Enqueue a single `index` job for a list of document IDs (re-indexing pipeline).

---

## Runtime RAG API (`/rag`)

Defined in `app/api/v1/routes/rag.py`:

- **POST `/rag/generate/questions`**
  - Body: `RAGGenerateRequest` (board, class, subject, chapter_number, difficulty, question_type, count).
  - Calls `rag.query.generate_questions.generate_questions` which:
    - Queries Qdrant for relevant chunks.
    - Calls the LLM with a grounded prompt.
    - Returns structured questions.
  - Response: `RAGGenerateResponse` with list of `RAGQuestionOut`:
    - `stem`, `options`, `answer`, `qtype`, `difficulty`.

This path is essentially a **runtime RAG** question-generation API that uses the existing Qdrant index but bypasses the pool/worker system.

---

## Niyukti Compatibility API (`/api/v1`)

Mounted in `main.py` as:

- `app.include_router(niyukti_compat_router, prefix="/api/v1", tags=["niyukti-compat"])`

Routes are defined in `app/api/v1/routes/niyukti_compat.py` and provide a façade tailored to the Niyukti platform:

- **Diagnostic / Test Endpoints**
  - **GET `/api/v1/test-simple`** – simple health probe.
  - **POST `/api/v1/test-post`** – echoes method/path.
  - **GET `/api/v1/test-error-handling`** – triggers a controlled error to test error handling.

- **Ingestion (Niyukti-compatible)**
  - **POST `/api/v1/ingest`**
  - **POST `/api/v1/ingest/pdf`** (alias)
    - Wraps the core `ingest_pdf` implementation.
    - Accepts multipart PDF and minimal metadata (board, class_level, subject, chapter info, `doc_id`).
    - Derives `tenant_id` from header `X-Tenant-ID` or defaults to `"niyukti-platform"`.
    - Returns same `PDFIngestResponse` as `/v1/ingest/pdf`.

  - **GET `/api/v1/ingest/stats`**
    - Returns ingestion statistics for the current tenant:
      - `corpus_id`, number of PDFs uploaded, chunks created, and last index job status.

- **Pool Status (Read-only)**
  - **GET `/api/v1/pool/status`**
    - Query params: `class_level`, `subject`, optional `chapter`, plus `syllabus`, `syllabus_version` (defaults: CBSE/2024).
    - Headers: `X-Corpus-ID` (required), `X-Tenant-ID` (optional).
    - Reads pool status across difficulties by:
      - Checking `QuestionPool` and `PoolQuestion` tables.
      - Looking at Redis-backed list lengths.
    - Returns a `PoolStatusResponse` summarizing:
      - `status`: `ready`, `ready_partial`, `pending`, `generating`, or `failed`.
      - Per-difficulty counts and last errors.
    - **Never triggers generation**, only introspection.

- **Question Serving (Niyukti contract)**
  - **POST `/api/v1/generate-questions`**
    - Request: `NiyuktiGenerateRequest` (class_level, subject, mcq/descriptive counts, optional per-difficulty counts and chapter, etc.).
    - Core behavior:
      - Resolves `corpus_id` from:
        - `X-Corpus-ID` header (preferred), or
        - metadata lookup in `GeneratedQuestion` & `Corpus`.
      - Computes a **difficulty mix** based on requested counts.
      - Performs **DB-only serving** from:
        - `PoolQuestion` joined with `QuestionPool` (pool-based), and
        - Fallback to `GeneratedQuestion` when pools are insufficient.
      - Supports incremental sync via header `X-After-Serial-No` (only returns questions with a higher serial number).
      - Returns:
        - If enough questions are available:
          - A `NiyuktiGenerateResponse` with a list of `NiyuktiQuestionResponse` items, properly mapped to MCQ or descriptive format.
        - If pools are not yet sufficient:
          - HTTP 202 with body `{ status: "generating", ... }`, signaling the frontend to retry later.

This compatibility layer lets Niyukti consume questions without being aware of internal pools, jobs, or schemas.

---

## Health & Observability

- **Health Endpoints**
  - `/healthz`, `/health` (top-level and under `/v1`):
    - Exposes basic runtime info, key tails (for debugging, not full secrets), LLM client type, and AgentBill status.

- **Metrics**
  - `metrics_router` is included; each request passes through `_metrics_mw` which:
    - Measures latency.
    - Records per-request metrics via `observe_request`.

- **Logging**
  - `app/observability/logging.py` configures structured logging.
  - Many endpoints log key events (ingest, admin jobs, pool serving, compatibility flows).

- **Rate Limiting**
  - `slowapi` with default limit `requests_per_minute` from settings.

---

## Using This Service as a Microservice (From Another Project)

This project is already structured as an HTTP microservice. Another project (backend or frontend) can integrate by:

- Deploying this service (e.g. behind a load balancer or API gateway).
- Using HTTPS to call exposed endpoints as per the contracts above.
- Supplying an API key (if `API_KEY`/`API_KEYS` is configured) via an `X-API-Key` or configured header (per `APIKeyMiddleware` behavior).

### Typical Integration Flow

**1. Health & Configuration**
- **GET** `http(s)://<service-host>/healthz`
  - Verify service is up and check basic runtime configuration.

**2. Corpus and Document Setup (generic integration via `/v1`)**
- **Create a corpus** for your tenant:
  - `POST /v1/corpora` with `CorpusCreateRequest`.
- **Upload or register PDFs**:
  - Cloud/R2 mode: `POST /v1/corpora/{corpus_id}/documents`.
  - Local mode: `POST /v1/corpora/{corpus_id}/documents/register`.
- **Start indexing**:
  - `POST /v1/corpora/{corpus_id}/index-jobs` with optional `document_ids`.
  - Poll `GET /v1/jobs/{job_id}` until status is `completed`.

**3. Admin-side Question Generation (internal or trusted caller)**
- For each document (or batch of documents):
  - Call **`POST /v1/admin/generate-questions`** or **`POST /v1/admin/generate-questions/batch`** to enqueue LLM-based generation jobs.
  - Optionally:
    - Monitor `GeneratedQuestion` via `GET /v1/admin/generated-questions`.
    - Run `POST /v1/admin/prewarm-pools` to promote questions into pools and optimize serving.

**4. Question Serving Paths**

- **Pool-based (fast path, internal)**:
  - Your orchestrator calls:
    - `POST /v1/question-generation/pools` with `PoolRequest`.
    - Must include header: `X-Internal-Request: true`.
  - Behavior:
    - If pools are ready: returns questions immediately.
    - If not: returns `status="generating"`; orchestrator decides whether to fall back or wait.

- **Niyukti-style (external)**
  - Another project (e.g. Niyukti’s backend/frontend) calls:
    - **Ingest**: `POST /api/v1/ingest` / `/api/v1/ingest/pdf`.
    - **Check ingest stats**: `GET /api/v1/ingest/stats`.
    - **Check pool status**: `GET /api/v1/pool/status`.
    - **Serve questions**: `POST /api/v1/generate-questions`.
  - This yields a very simple contract:
    - Provide class, subject, and counts.
    - Receive questions, or a `generating` status to signal that generation is in progress.

- **RAG-based (on-demand)**
  - If you want **runtime RAG** without pre-generated pools:
    - Call `POST /rag/generate/questions` with `RAGGenerateRequest`.
    - Use when:
      - You have high-latency budgets, and
      - Want more flexible, on-the-fly generation.

---

## Internal Call Flows (End-to-End)

### A. Ingest + Index + Generate + Serve (Pool-based)

1. **Ingest**
   - Client (Niyukti or other) uploads a PDF:
     - Via `/api/v1/ingest` (Niyukti) or `/v1/ingest/pdf` (generic).
   - Service:
     - Validates metadata and PDF header.
     - Uploads PDF to R2.
     - Creates `Document` and an `ingest` job.
     - Creates an `index` job for Qdrant.

2. **Indexing Worker**
   - Background worker picks up `index` job:
     - Downloads PDF via storage backend.
     - Parses into chunks (`app/indexing/pdf_parse.py`, `app/indexing/chunker.py`).
     - Embeds chunks via `app/embeddings/*`.
     - Upserts into Qdrant (`app/indexing/qdrant_pipeline.py`).
     - Marks job as completed.

3. **Admin Generation**
   - Admin service (internal to your platform) decides what questions to pre-generate:
     - Calls `/v1/admin/generate-questions` or `/v1/admin/generate-questions/batch`.
   - Worker:
     - Picks up `admin_generate_questions` jobs.
     - Runs retrieval over Qdrant.
     - Calls LLM via `app/llm/factory.py`.
     - Validates and stores `GeneratedQuestion` and/or `PoolQuestion`.

4. **Pool Prewarm**
   - Admin calls `/v1/admin/prewarm-pools` to promote questions into `QuestionPool`/Redis.
   - Pools are now ready for low-latency serving.

5. **Serving**
   - Internal orchestrator calls `/v1/question-generation/pools` with `PoolRequest`:
     - If enough questions present: returns them immediately.
     - Otherwise: returns `status="generating"` and pool metadata.
   - Niyukti frontends can alternatively call `/api/v1/generate-questions`, which:
     - Reads from `PoolQuestion` / `GeneratedQuestion` directly.
     - Returns questions in a Niyukti-specific schema (MCQ/descriptive).

### B. Runtime RAG Flow

1. PDFs are ingested and indexed to Qdrant as in steps 1–2 above.
2. Client calls `POST /rag/generate/questions`:
   - Qdrant is queried directly based on (board, class, subject, chapter, difficulty, question_type, count).
   - LLM is called with grounded context.
   - Questions are returned directly in `RAGGenerateResponse`.

---

## Summary

- This project is a **self-contained FastAPI microservice** for question generation and serving.
- It exposes:
  - A **versioned v1 API** (`/v1`) for generic integrations and admin workflows.
  - A **Niyukti-compatibility API** (`/api/v1`) for drop-in integration with existing Niyukti flows.
  - A **RAG API** (`/rag`) for direct, on-the-fly question generation over Qdrant.
- Another project can integrate simply by:
  - Deploying this service.
  - Calling its HTTP endpoints with the described payloads.
  - Treating it as a pure **microservice over HTTP**, with persistence handled entirely within this repo (Postgres, Qdrant, Redis, R2, and LLM backends).


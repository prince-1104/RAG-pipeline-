from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.v1.schemas.generation import GenerateRequest, GenerateResponse
from app.api.v1.schemas.ingest import PDFIngestResponse
from app.api.v1.schemas.niyukti_compat import (
    NiyuktiGenerateRequest,
    NiyuktiGenerateResponse,
    NiyuktiQuestionResponse,
)
from app.db.models import Corpus
from app.db.session import get_session
from app.generation.sync import generate_questions_sync
from app.settings import get_settings

# Import ingest functionality
from app.api.v1.routes.ingest import ingest_pdf


router = APIRouter()


@router.get("/test-simple")
def test_simple():
    """Simple test endpoint to verify routing works"""
    return {"status": "ok", "message": "Endpoint is working"}


@router.post("/test-post")
def test_post(request: Request):
    """Test POST endpoint"""
    return {"status": "ok", "method": request.method, "path": str(request.url.path)}


@router.get("/test-error-handling")
def test_error_handling():
    """Test endpoint to verify error handling works"""
    try:
        raise ValueError("Test error message")
    except Exception as e:
        from fastapi.responses import ORJSONResponse
        from app.observability.logging import log
        log.error("test_error", error=str(e))
        return ORJSONResponse(
            status_code=500,
            content={"detail": f"test_error: {str(e)}", "error_type": type(e).__name__}
        )


@router.post("/ingest", response_model=PDFIngestResponse)
async def ingest_pdf_niyukti(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    doc_id: str = Form(..., description="Document ID provided by caller"),
    board: str = Form(..., description="Board identifier (e.g., 'CBSE')"),
    class_level: int = Form(..., ge=1, le=12, description="Class level (1-12)"),
    subject: str = Form(..., description="Subject name"),
):
    """
    PDF ingestion endpoint for Niyukti platform compatibility.
    This endpoint is mounted at /api/v1/ingest to match frontend expectations.
    
    Automatically triggers indexing pipeline:
    - Uploads PDF to R2
    - Creates Document record
    - Enqueues index job (chunking → embedding → FAISS index)
    
    Returns index_job_id for tracking progress via GET /api/v1/jobs/{index_job_id}
    """
    from app.observability.logging import log
    
    # Get tenant_id from header or use default
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    log.info(
        "pdf_ingest_niyukti_endpoint_called",
        doc_id=doc_id,
        board=board,
        class_level=class_level,
        subject=subject,
        tenant_id=tenant_id,
        filename=file.filename if file else None,
        path=str(request.url.path),
    )
    
    # Delegate to the main ingest implementation with tenant_id
    return await ingest_pdf(file, doc_id, board, class_level, subject, tenant_id)


@router.get("/ingest/stats")
def get_ingest_stats(request: Request):
    """
    Get ingestion statistics for Niyukti UI.
    Returns counts of PDFs uploaded, chunks created, and index job status.
    """
    from sqlalchemy import func
    from sqlmodel import select
    from app.db.models import Document, Chunk, GenerationJob, JobKind, JobStatus
    
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    with get_session() as session:
        # Find corpus for this tenant
        corpus = session.exec(
            select(Corpus).where(Corpus.tenant_id == tenant_id).limit(1)
        ).first()
        
        if not corpus:
            return {
                "tenant_id": tenant_id,
                "corpus_id": None,
                "pdfs_uploaded": 0,
                "chunks_created": 0,
                "last_index_job": None,
            }
        
        # Count documents
        pdf_count = session.exec(
            select(func.count(Document.id)).where(Document.corpus_id == corpus.id)
        ).one() or 0
        
        # Count chunks
        chunk_count = session.exec(
            select(func.count(Chunk.id)).where(Chunk.corpus_id == corpus.id)
        ).one() or 0
        
        # Get last index job
        last_index_job = session.exec(
            select(GenerationJob)
            .where(
                GenerationJob.corpus_id == corpus.id,
                GenerationJob.kind == JobKind.index,
            )
            .order_by(GenerationJob.created_at.desc())
            .limit(1)
        ).first()
        
        last_job_info = None
        if last_index_job:
            import json
            result = None
            if last_index_job.result_json:
                try:
                    result = json.loads(last_index_job.result_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            last_job_info = {
                "job_id": last_index_job.id,
                "status": last_index_job.status.value,
                "created_at": last_index_job.created_at.isoformat() if last_index_job.created_at else None,
                "started_at": last_index_job.started_at.isoformat() if last_index_job.started_at else None,
                "finished_at": last_index_job.finished_at.isoformat() if last_index_job.finished_at else None,
                "progress": result,
            }
        
        return {
            "tenant_id": tenant_id,
            "corpus_id": corpus.id,
            "pdfs_uploaded": pdf_count,
            "chunks_created": chunk_count,
            "last_index_job": last_job_info,
        }


def _map_subject_name(subject: str) -> str:
    """Map Niyukti subject names to our format"""
    subject_mapping = {
        "maths": "Mathematics",
        "math": "Mathematics",
        "science": "Science",
        "english": "English",
        "hindi": "Hindi",
        "social_science": "Social Science",
        "socialscience": "Social Science",
        "sst": "Social Science",
    }
    # Try exact match first
    if subject in subject_mapping:
        return subject_mapping[subject]
    # Try case-insensitive match
    subject_lower = subject.lower()
    for key, value in subject_mapping.items():
        if subject_lower == key.lower():
            return value
    # Return capitalized version if no mapping found
    return subject.capitalize()


def _convert_niyukti_to_blueprint(req: NiyuktiGenerateRequest, tenant_id: str, corpus_id: str) -> GenerateRequest:
    """Convert Niyukti's simple request format to our blueprint format"""
    settings = get_settings()
    
    # Map subject name
    mapped_subject = _map_subject_name(req.subject)
    # #region agent log
    try:
        import json
        from app.debug_log import DEBUG_LOG_PATH, ensure_log_dir
        ensure_log_dir()
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"niyukti_compat.py:109","message":"Subject name mapping","data":{"original_subject":req.subject,"mapped_subject":mapped_subject,"class_level":req.class_level,"corpus_id":corpus_id},"timestamp":int(__import__("time").time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion

    # Best-effort: use all known chapters for this (corpus,class,subject) so
    # retrieval doesn't fail when Niyukti doesn't pass chapter names.
    chapters: list[str] = ["Unknown Chapter"]
    try:
        from sqlmodel import select
        from app.db.models import Chunk

        with get_session() as session:
            rows = session.exec(
                select(Chunk.chapter)
                .where(
                    Chunk.corpus_id == corpus_id,
                    Chunk.class_level == req.class_level,
                    Chunk.subject == mapped_subject,
                )
                .distinct()
            ).all()
        chap = [str(r).strip() for r in rows if r and str(r).strip()]
        # #region agent log
        try:
            import json
            from app.debug_log import DEBUG_LOG_PATH
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"niyukti_compat.py:128","message":"Chapter lookup result","data":{"chapters_found":len(chap),"chapters":chap[:10],"mapped_subject":mapped_subject,"class_level":req.class_level},"timestamp":int(__import__("time").time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        if chap:
            chapters = chap[:200]  # keep it bounded
    except Exception as e:
        # #region agent log
        try:
            import json
            from app.debug_log import DEBUG_LOG_PATH
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"niyukti_compat.py:132","message":"Chapter lookup exception","data":{"error":str(e)[:200],"mapped_subject":mapped_subject},"timestamp":int(__import__("time").time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        chapters = ["Unknown Chapter"]
    
    # Determine question types and counts
    question_types = []
    difficulty_mix = {"easy": 0, "medium": 0, "hard": 0}
    
    # Check if per-difficulty counts are provided (from Niyukti UI)
    has_difficulty_counts = req.easy_count is not None or req.medium_count is not None or req.hard_count is not None
    
    if has_difficulty_counts:
        # Use per-difficulty counts from UI
        easy_total = req.easy_count or 0
        medium_total = req.medium_count or 0
        hard_total = req.hard_count or 0
        total_count = easy_total + medium_total + hard_total
        
        # Apply MCQ ratio to each difficulty level if provided
        mcq_ratio = req.mcq_ratio if req.mcq_ratio is not None else 70.0  # Default 70% if not specified
        mcq_ratio = max(0, min(100, mcq_ratio)) / 100.0  # Normalize to 0-1
        
        # Distribute each difficulty level by MCQ ratio
        for difficulty, count in [("easy", easy_total), ("medium", medium_total), ("hard", hard_total)]:
            if count > 0:
                mcq_count_for_diff = int(count * mcq_ratio)
                desc_count_for_diff = count - mcq_count_for_diff
                
                if mcq_count_for_diff > 0:
                    if "mcq" not in question_types:
                        question_types.append("mcq")
                    difficulty_mix[difficulty] += mcq_count_for_diff
                
                if desc_count_for_diff > 0:
                    if "short" not in question_types:
                        question_types.append("short")
                    difficulty_mix[difficulty] += desc_count_for_diff
    else:
        # Fallback to original logic: use mcq_count/descriptive_count
        total_count = req.mcq_count + req.descriptive_count
        
        # If difficulty_level is specified, use it; otherwise distribute automatically
        if req.difficulty_level and req.difficulty_level.lower() in ["easy", "medium", "hard"]:
            # Use specified difficulty level for all questions
            difficulty = req.difficulty_level.lower()
            if req.mcq_count > 0:
                question_types.append("mcq")
                difficulty_mix[difficulty] += req.mcq_count
            if req.descriptive_count > 0:
                question_types.append("short")  # Map "descriptive" to "short"
                difficulty_mix[difficulty] += req.descriptive_count
        else:
            # Auto-distribute difficulty (original logic)
            if req.mcq_count > 0:
                question_types.append("mcq")
                # Distribute MCQ difficulty: 40% easy, 40% medium, 20% hard
                easy_mcq = max(1, int(req.mcq_count * 0.4))
                medium_mcq = max(1, int(req.mcq_count * 0.4))
                hard_mcq = req.mcq_count - easy_mcq - medium_mcq
                difficulty_mix["easy"] += easy_mcq
                difficulty_mix["medium"] += medium_mcq
                difficulty_mix["hard"] += max(0, hard_mcq)
            
            if req.descriptive_count > 0:
                question_types.append("short")  # Map "descriptive" to "short"
                # Distribute descriptive difficulty: 30% easy, 50% medium, 20% hard
                easy_desc = max(1, int(req.descriptive_count * 0.3))
                medium_desc = max(1, int(req.descriptive_count * 0.5))
                hard_desc = req.descriptive_count - easy_desc - medium_desc
                difficulty_mix["easy"] += easy_desc
                difficulty_mix["medium"] += medium_desc
                difficulty_mix["hard"] += max(0, hard_desc)
    
    # Ensure at least one question type
    if not question_types:
        question_types = ["mcq"]
        difficulty_mix = {"easy": 1, "medium": 0, "hard": 0}
        total_count = 1
    
    # Build blueprint
    from app.api.v1.schemas.generation import Blueprint, DifficultyMix, QuestionConstraints
    
    blueprint = Blueprint(
        blueprint_version="2026-01-02",
        job_summary={
            "skills": ["problem-solving", "conceptual understanding"],
            "role": "student assessment",
            "focus_topics": [mapped_subject.lower()],
        },
        constraints=QuestionConstraints(
            class_level=req.class_level,
            subject=mapped_subject,
            chapters=chapters,
            language="en",
        ),
        question_types=question_types,
        count=total_count,
        difficulty_mix=DifficultyMix(**difficulty_mix),
        coverage_strategy="balanced",
    )
    
    return GenerateRequest(
        tenant_id=tenant_id,
        corpus_id=corpus_id,
        blueprint=blueprint,
        max_context_chunks=settings.max_context_chunks,
    )


def _convert_pool_questions_to_niyukti_format(
    questions: list[dict], class_level: int, subject: str
) -> NiyuktiGenerateResponse:
    """Convert pool questions (dict format) to Niyukti format"""
    niyukti_questions = []
    
    for q in questions:
        # Map question type
        qtype = q.get("qtype", "mcq")
        question_type = "mcq" if qtype == "mcq" else "descriptive"
        
        # Build question response
        niyukti_q = NiyuktiQuestionResponse(
            question_text=q.get("stem", ""),
            question_type=question_type,
            class_level=class_level,
            subject=subject.lower(),
        )
        
        if question_type == "mcq":
            niyukti_q.options = q.get("options") or []
            niyukti_q.correct_answer = q.get("answer") or ""
        else:
            niyukti_q.expected_answer_text = q.get("answer") or ""
            niyukti_q.correct_answer = q.get("answer") or ""
            if q.get("explanation"):
                keywords = [
                    word.lower()
                    for word in q.get("explanation", "").split()
                    if len(word) > 4 and word.isalnum()
                ][:5]
                niyukti_q.expected_keywords = keywords
        
        niyukti_questions.append(niyukti_q)
    
    return NiyuktiGenerateResponse(questions=niyukti_questions)


def _trigger_pool_generation(
    corpus_id: str,
    syllabus: str,
    class_level: int,
    subject: str,
    difficulty: str,
    syllabus_version: str,
) -> None:
    """Trigger background job to generate question pool (LLM called in worker)."""
    import json
    from app.db.models import GenerationJob, JobKind, JobStatus
    from app.observability.logging import log
    
    with get_session() as session:
        # Check if job already exists
        from sqlmodel import select
        # Now that enum values are lowercase, direct comparison should work
        existing_jobs = session.exec(
            select(GenerationJob).where(
                GenerationJob.kind == JobKind.generate_question_pool,
                GenerationJob.corpus_id == corpus_id,
                GenerationJob.status.in_([JobStatus.queued, JobStatus.running]),
            )
        ).all()
        
        # Check if there's already a queued/running job for this exact pool
        for job in existing_jobs:
            if job.status in [JobStatus.queued, JobStatus.running]:
                try:
                    job_data = json.loads(job.request_json)
                    if (job_data.get("syllabus") == syllabus and
                        job_data.get("class_level") == class_level and
                        job_data.get("subject") == subject and
                        job_data.get("difficulty") == difficulty and
                        job_data.get("syllabus_version") == syllabus_version):
                        log.info("pool_generation_job_already_queued", job_id=job.id)
                        return
                except Exception:
                    continue
        
        # Create new background pool generation job
        job_data = {
            "corpus_id": corpus_id,
            "syllabus": syllabus,
            "class_level": class_level,
            "subject": subject,
            "difficulty": difficulty,
            "syllabus_version": syllabus_version,
        }
        
        # Create job - use enum value directly to avoid SQLAlchemy serialization issues
        # SQLAlchemy with native_enum=False should use enum.value, but we'll ensure it explicitly
        job = GenerationJob(
            kind=JobKind.generate_question_pool,
            status=JobStatus.queued,
            tenant_id="system",
            corpus_id=corpus_id,
            request_json=json.dumps(job_data),
        )
        session.add(job)
        session.commit()
        log.info("pool_generation_job_created", job_id=job.id, difficulty=difficulty)


def _convert_response_to_niyukti_format(
    response: GenerateResponse, class_level: int, subject: str
) -> NiyuktiGenerateResponse:
    """Convert our response format to Niyukti's expected format"""
    niyukti_questions = []
    
    for q in response.questions or []:
        # Map question type
        question_type = "mcq" if q.qtype == "mcq" else "descriptive"
        
        # Build question response
        niyukti_q = NiyuktiQuestionResponse(
            question_text=q.stem,
            question_type=question_type,
            class_level=class_level,
            subject=subject.lower(),
        )
        
        if question_type == "mcq":
            # MCQ format
            niyukti_q.options = q.options or []
            niyukti_q.correct_answer = q.answer or ""
        else:
            # Descriptive format
            niyukti_q.expected_answer_text = q.answer or ""
            niyukti_q.correct_answer = q.answer or ""  # Also set correct_answer for compatibility
            # Extract keywords from explanation (simple approach)
            if q.explanation:
                # Simple keyword extraction - split by common words
                keywords = [
                    word.lower()
                    for word in q.explanation.split()
                    if len(word) > 4 and word.isalnum()
                ][:5]  # Limit to 5 keywords
                niyukti_q.expected_keywords = keywords
        
        niyukti_questions.append(niyukti_q)
    
    return NiyuktiGenerateResponse(questions=niyukti_questions)


@router.post("/generate-questions")
def generate_questions_niyukti(request: Request, req: NiyuktiGenerateRequest):
    return _generate_questions_niyukti_impl(request, req)


def _generate_questions_niyukti_impl(request: Request, req: NiyuktiGenerateRequest):
    """
    Compatibility endpoint for Niyukti platform.
    Expected: POST /api/v1/generate-questions; Request: { class_level, subject, mcq_count, descriptive_count }; Response: { questions: [...] }
    """
    import json
    import sys
    import traceback
    from fastapi.responses import JSONResponse

    # #region agent log
    try:
        from app.debug_log import DEBUG_LOG_PATH, ensure_log_dir
        ensure_log_dir()
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"handler","location":"niyukti_compat:_generate_questions_niyukti_impl","message":"generate-questions handler entered","data":{"class_level":req.class_level,"subject":req.subject},"timestamp":int(__import__("time").time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    try:
        from app.observability.logging import log

        # Resolve corpus: X-Corpus-ID takes priority over tenant lookup
        tenant_id = request.headers.get("X-Tenant-ID")
        corpus_id = request.headers.get("X-Corpus-ID")

        if corpus_id:
            with get_session() as session:
                corpus = session.get(Corpus, corpus_id)
                if not corpus:
                    raise HTTPException(status_code=404, detail="corpus_not_found")
                # Do NOT reject on tenant mismatch when X-Corpus-ID is provided
                if tenant_id and corpus.tenant_id != tenant_id:
                    log.warning(
                        "tenant_mismatch_ignored_for_corpus_id",
                        tenant_id=tenant_id,
                        corpus_id=corpus_id,
                        corpus_tenant_id=corpus.tenant_id,
                    )
        else:
            if not tenant_id:
                raise HTTPException(status_code=400, detail="missing_corpus_id_or_tenant_id")
            # Legacy fallback: resolve corpus by tenant_id
            with get_session() as session:
                from sqlmodel import select

                corpus = session.exec(
                    select(Corpus).where(Corpus.tenant_id == tenant_id).limit(1)
                ).first()

                if not corpus:
                    raise HTTPException(
                        status_code=404,
                        detail="no_corpus_found_for_tenant_please_create_corpus_first",
                    )
                corpus_id = corpus.id
        
        # PRODUCTION FLOW: Redis → DB → Background Job (NO LLM on user request)
        mapped_subject = _map_subject_name(req.subject)
        syllabus = "CBSE"  # Default syllabus, can be made configurable
        syllabus_version = "2024"  # Default version
        
        # Determine difficulty mix from request
        difficulty_mix = {}
        if req.easy_count is not None:
            difficulty_mix["easy"] = req.easy_count or 0
        if req.medium_count is not None:
            difficulty_mix["medium"] = req.medium_count or 0
        if req.hard_count is not None:
            difficulty_mix["hard"] = req.hard_count or 0
        
        # Fallback to auto-distribution if no difficulty counts provided
        if not difficulty_mix:
            total = req.mcq_count + req.descriptive_count
            if total > 0:
                difficulty_mix = {
                    "easy": max(1, int(total * 0.4)),
                    "medium": max(1, int(total * 0.4)),
                    "hard": max(0, total - int(total * 0.4) - int(total * 0.4)),
                }
        
        # STEP A: Check Redis (hot path)
        all_questions = []
        pool_info = {}
        needs_generation = False
        
        for difficulty, count in difficulty_mix.items():
            if count <= 0:
                continue
            
            # Try to get questions from Redis pool
            from app.pools.service import get_questions_from_pool, get_pool_status, get_or_create_pool
            
            redis_questions = get_questions_from_pool(
                syllabus=syllabus,
                class_level=req.class_level,
                subject=mapped_subject,
                difficulty=difficulty,
                syllabus_version=syllabus_version,
                count=count,
            )
            
            if len(redis_questions) >= count:
                # STEP A SUCCESS: Redis has questions
                pool = get_or_create_pool(syllabus, req.class_level, mapped_subject, difficulty, syllabus_version)
                pool_info[difficulty] = pool.id
                
                # Convert PoolQuestion to Niyukti format
                for pq in redis_questions[:count]:
                    q_data = json.loads(pq.question_json)
                    all_questions.append(q_data)
            else:
                # STEP B: Check DB (cold but safe)
                pool_status = get_pool_status(syllabus, req.class_level, mapped_subject, difficulty, syllabus_version)
                
                if pool_status == "ready":
                    # DB has active pool but Redis is empty - rebuild Redis async
                    pool = get_or_create_pool(syllabus, req.class_level, mapped_subject, difficulty, syllabus_version)
                    pool_info[difficulty] = pool.id
                    
                    # Trigger async Redis rebuild
                    from app.pools.rebuild import rebuild_redis_pool_async
                    import threading
                    threading.Thread(target=rebuild_redis_pool_async, args=(pool.id,), daemon=True).start()
                    
                    # Fetch from DB while Redis rebuilds
                    from sqlmodel import select
                    from app.db.models import PoolQuestion
                    
                    with get_session() as session:
                        db_questions = session.exec(
                            select(PoolQuestion)
                            .where(PoolQuestion.pool_id == pool.id)
                            .limit(count)
                        ).all()
                        
                        for pq in db_questions[:count]:
                            q_data = json.loads(pq.question_json)
                            all_questions.append(q_data)
                else:
                    # STEP C: Trigger background job (ONLY enqueue)
                    needs_generation = True
                    pool = get_or_create_pool(syllabus, req.class_level, mapped_subject, difficulty, syllabus_version)
                    pool_info[difficulty] = pool.id
                    
                    # Trigger background pool generation
                    _trigger_pool_generation(corpus_id, syllabus, req.class_level, mapped_subject, difficulty, syllabus_version)
        
        # Return response
        if all_questions and not needs_generation:
            # Questions available - return immediately
            niyukti_response = _convert_pool_questions_to_niyukti_format(
                all_questions, req.class_level, req.subject
            )
            return niyukti_response
        else:
            # Need to generate - return 202 Accepted
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=202,
                content={
                    "status": "generating",
                    "message": "questions_are_being_generated_please_check_back_shortly",
                    "pool_info": pool_info,
                    "corpus_id": corpus_id,
                    "class_level": req.class_level,
                    "subject": req.subject,
                }
            )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions as-is (they'll be caught by handler)
        raise http_exc
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = str(e)[:500]
        print(f"ERROR in generate_questions_niyukti: {type(e).__name__}: {error_msg}", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        sys.stderr.flush()
        return JSONResponse(
            status_code=500,
            content={"detail": f"generation_failed: {error_msg}", "error_type": type(e).__name__}
        )

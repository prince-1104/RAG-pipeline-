from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.api.v1.schemas.admin import (
    AdminBatchGenerateQuestionsRequest,
    AdminBatchGenerateQuestionsResponse,
    AdminGeneratedQuestionOut,
    AdminGeneratedQuestionsResponse,
    AdminGenerateQuestionsRequest,
    AdminGenerateQuestionsResponse,
    AdminPrewarmQuestionsResponse,
    AdminPrewarmPoolsResponse,
    AdminReindexRequest,
    AdminReindexResponse,
    DocumentJobStatus,
)
from app.db.models import Corpus, Document, GeneratedQuestion, GenerationJob, JobKind, JobStatus
from app.db.session import get_session
from app.observability.logging import log
from app.pools.service import (
    fail_prewarm_run,
    finish_prewarm_run,
    get_prewarm_promoted_questions,
    invalidate_pools_for_syllabus,
    promote_all_generated_questions_to_pools,
    start_prewarm_run,
)


router = APIRouter()


@router.get("/generated-questions", response_model=AdminGeneratedQuestionsResponse)
def admin_get_generated_questions(
    generation_job_id: str | None = None,
    document_id: str | None = None,
    class_level: int | None = None,
    subject: str | None = None,
    chapter: str | None = None,
    difficulty: str | None = None,
    qtype: str | None = None,
    language: str | None = None,
):
    with get_session() as session:
        stmt = select(GeneratedQuestion)
        if generation_job_id:
            stmt = stmt.where(GeneratedQuestion.job_id == generation_job_id)
        if document_id:
            stmt = stmt.where(GeneratedQuestion.document_id == document_id)
        if class_level is not None:
            stmt = stmt.where(GeneratedQuestion.class_level == class_level)
        if subject:
            stmt = stmt.where(GeneratedQuestion.subject == subject)
        if chapter:
            stmt = stmt.where(GeneratedQuestion.chapter == chapter)
        if difficulty:
            stmt = stmt.where(GeneratedQuestion.difficulty == difficulty)
        if qtype:
            stmt = stmt.where(GeneratedQuestion.qtype == qtype)
        if language:
            stmt = stmt.where(GeneratedQuestion.language == language)

        rows = session.exec(
            stmt.order_by(GeneratedQuestion.created_at, GeneratedQuestion.id)
        ).all()

    questions: list[AdminGeneratedQuestionOut] = []
    for row in rows:
        try:
            question_payload = json.loads(row.question_json) if row.question_json else {}
        except (TypeError, json.JSONDecodeError):
            question_payload = {}
        questions.append(
            AdminGeneratedQuestionOut(
                id=str(row.id),
                class_level=int(row.class_level),
                subject=str(row.subject),
                chapter=str(row.chapter),
                chapter_number=row.chapter_number,
                chapter_name=row.chapter_name,
                difficulty=str(row.difficulty),
                qtype=str(row.qtype),
                language=str(row.language),
                question_json=question_payload if isinstance(question_payload, dict) else {},
                created_at=row.created_at,
            )
        )
    return AdminGeneratedQuestionsResponse(questions=questions)


@router.post("/generate-questions", response_model=AdminGenerateQuestionsResponse)
def admin_generate_questions(request: Request, req: AdminGenerateQuestionsRequest):
    """
    Admin endpoint to enqueue pool-generation intent for uploaded PDFs.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    - PDF upload must NEVER trigger LLM
    - Institute APIs must NEVER enqueue generation jobs
    - Admin endpoints are asynchronous and return immediately
    
    The admin can specify:
    - Document ID (PDF that was uploaded)
    - Class range (from_class to to_class)
    - Subject
    - Difficulty level
    - Number of questions
    - Question types (mcq, short, long)
    - Optional chapters filter
    
    The intent is stored and expanded into pool-generation jobs by the worker.
    """
    # Get tenant_id from header (admin should be authenticated)
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    # Validate document exists
    with get_session() as session:
        document = session.get(Document, req.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="document_not_found")
        
        # Get corpus
        corpus = session.get(Corpus, document.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        
        # Validate tenant matches
        if corpus.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")
        
        # Validate class range
        if req.from_class > req.to_class:
            raise HTTPException(
                status_code=400,
                detail="from_class_must_be_less_than_or_equal_to_to_class"
            )
        
        # Validate document has required metadata
        if not document.class_level or not document.subject:
            raise HTTPException(
                status_code=400,
                detail="document_missing_required_metadata_class_level_and_subject_required"
            )
        
        # Create admin generation job
        # Generate questions for each class in the range
        job_data = {
            "document_id": req.document_id,
            "corpus_id": document.corpus_id,
            "from_class": req.from_class,
            "to_class": req.to_class,
            "subject": req.subject,
            "difficulty": req.difficulty,
            "question_count": req.question_count,
            "question_types": req.question_types,
            "chapters": req.chapters,
            "syllabus": document.syllabus or "CBSE",
            "syllabus_version": document.syllabus_version or "2024",
        }
        
        job = GenerationJob(
            kind=JobKind.admin_generate_questions,
            status=JobStatus.queued,
            tenant_id=tenant_id,
            corpus_id=document.corpus_id,
            request_json=json.dumps(job_data),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        
        log.info(
            "admin_question_generation_job_created",
            job_id=job.id,
            document_id=req.document_id,
            from_class=req.from_class,
            to_class=req.to_class,
            subject=req.subject,
            difficulty=req.difficulty,
            question_count=req.question_count,
            tenant_id=tenant_id,
        )
        
        return AdminGenerateQuestionsResponse(
            job_id=job.id,
            status="queued",
            message=f"Question generation job queued for document {req.document_id}. Generating {req.question_count} {req.difficulty} questions for classes {req.from_class}-{req.to_class} in {req.subject}.",
        )


@router.post("/generate-questions/batch", response_model=AdminBatchGenerateQuestionsResponse)
def admin_batch_generate_questions(request: Request, req: AdminBatchGenerateQuestionsRequest):
    """
    Batch admin endpoint to trigger question generation for multiple uploaded PDFs.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    - PDF upload must NEVER trigger LLM
    - Institute APIs must NEVER enqueue generation jobs
    - Admin endpoints are asynchronous and return immediately
    
    Supports:
    - Multiple document IDs (batch processing)
    - "Any" difficulty (generates for all difficulty levels)
    - "Descriptive" question type (maps to "short" and "long")
    
    Returns per-document job status for tracking progress.
    """
    # Get tenant_id from header (admin should be authenticated)
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    # Map "descriptive" to ["short", "long"]
    mapped_question_types = []
    for qt in req.question_types:
        if qt == "descriptive":
            mapped_question_types.extend(["short", "long"])
        else:
            mapped_question_types.append(qt)
    # Remove duplicates while preserving order
    mapped_question_types = list(dict.fromkeys(mapped_question_types))
    
    # Determine difficulty levels to generate
    if req.difficulty == "any":
        difficulty_levels = ["easy", "medium", "hard"]
    else:
        difficulty_levels = [req.difficulty]
    
    # Validate class range
    if req.from_class > req.to_class:
        raise HTTPException(
            status_code=400,
            detail="from_class_must_be_less_than_or_equal_to_to_class"
        )
    
    document_statuses: list[DocumentJobStatus] = []
    jobs_created = 0
    
    with get_session() as session:
        # Process each document
        for document_id in req.document_ids:
            try:
                # Validate document exists
                document = session.get(Document, document_id)
                if not document:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed_terminal",
                        message=f"Document {document_id} not found",
                        error="document_not_found"
                    ))
                    continue
                
                # Get corpus
                corpus = session.get(Corpus, document.corpus_id)
                if not corpus:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed_terminal",
                        message=f"Corpus not found for document {document_id}",
                        error="corpus_not_found"
                    ))
                    continue
                
                # Validate tenant matches
                if corpus.tenant_id != tenant_id:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed_terminal",
                        message=f"Tenant mismatch for document {document_id}",
                        error="tenant_mismatch"
                    ))
                    continue
                
                # Validate document has required metadata
                if not document.class_level or not document.subject:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed_terminal",
                        message=f"Document {document_id} missing required metadata (class_level and subject)",
                        error="missing_metadata"
                    ))
                    continue
                
                # Use document's subject if not provided, or validate it matches
                subject_to_use = req.subject
                if document.subject and document.subject != req.subject:
                    # Log warning but proceed with requested subject
                    log.warning(
                        "subject_mismatch",
                        document_id=document_id,
                        document_subject=document.subject,
                        requested_subject=req.subject,
                    )
                
                # Create jobs for each difficulty level
                job_ids = []
                for difficulty in difficulty_levels:
                    # Calculate questions per difficulty (distribute evenly)
                    questions_per_difficulty = max(1, req.question_count // len(difficulty_levels))
                    
                    job_data = {
                        "document_id": document_id,
                        "corpus_id": document.corpus_id,
                        "from_class": req.from_class,
                        "to_class": req.to_class,
                        "subject": subject_to_use,
                        "difficulty": difficulty,
                        "question_count": questions_per_difficulty,
                        "question_types": mapped_question_types,
                        "chapters": req.chapters,
                        "syllabus": document.syllabus or "CBSE",
                        "syllabus_version": document.syllabus_version or "2024",
                    }
                    
                    job = GenerationJob(
                        kind=JobKind.admin_generate_questions,
                        status=JobStatus.queued,
                        tenant_id=tenant_id,
                        corpus_id=document.corpus_id,
                        request_json=json.dumps(job_data),
                    )
                    session.add(job)
                    session.commit()
                    session.refresh(job)
                    job_ids.append(job.id)
                    jobs_created += 1
                    
                    log.info(
                        "admin_batch_question_generation_job_created",
                        job_id=job.id,
                        document_id=document_id,
                        from_class=req.from_class,
                        to_class=req.to_class,
                        subject=subject_to_use,
                        difficulty=difficulty,
                        question_count=questions_per_difficulty,
                        tenant_id=tenant_id,
                    )
                
                # Create status message
                if len(difficulty_levels) == 1:
                    message = f"Successfully queued generation of {req.question_count} {difficulty_levels[0]} questions for classes {req.from_class}-{req.to_class} in {subject_to_use}"
                else:
                    total_questions = req.question_count * len(difficulty_levels)
                    message = f"Successfully queued generation of {total_questions} questions ({req.question_count} per difficulty) for classes {req.from_class}-{req.to_class} in {subject_to_use}"
                
                document_statuses.append(DocumentJobStatus(
                    document_id=document_id,
                    job_id=job_ids[0] if job_ids else None,  # Return first job ID for tracking
                    status="queued",
                    message=message,
                ))
                
            except Exception as e:
                log.exception(
                    "admin_batch_generation_error",
                    document_id=document_id,
                    error=str(e),
                )
                document_statuses.append(DocumentJobStatus(
                    document_id=document_id,
                    job_id=None,
                    status="failed_terminal",
                    message=f"Error processing document {document_id}: {str(e)}",
                    error=str(e)[:200],  # Truncate error message
                ))
    
    return AdminBatchGenerateQuestionsResponse(
        total_documents=len(req.document_ids),
        jobs_created=jobs_created,
        documents=document_statuses,
        message=f"Batch processing complete. {jobs_created} job(s) created for {len([d for d in document_statuses if d.status == 'queued'])} document(s)."
    )


@router.post("/promote-generated-to-pool")
def admin_promote_generated_to_pool_deprecated():
    """
    Deprecated: per-document promotion is removed.
    """
    raise HTTPException(status_code=410, detail="deprecated_use_prewarm_pools")


@router.post("/prewarm-pool")
def admin_prewarm_pool_deprecated():
    """
    Deprecated: per-document promotion is removed.
    """
    raise HTTPException(status_code=410, detail="deprecated_use_prewarm_pools")


@router.post("/prewarm-pools", response_model=AdminPrewarmPoolsResponse)
def admin_prewarm_pools(request: Request):
    """
    GLOBAL pool prewarming endpoint.
    PROMOTION ONLY – NO GENERATION.
    """
    # Global operation: no request body required.
    syllabus = "CBSE"
    syllabus_version = "2024"
    run = start_prewarm_run(syllabus=syllabus, syllabus_version=syllabus_version)
    try:
        result = promote_all_generated_questions_to_pools(
            syllabus=syllabus,
            syllabus_version=syllabus_version,
            prewarm_id=run.id,
        )
        promoted_count = int(result.get("questions_promoted") or 0)
        pools_created = int(result.get("pools_created") or 0)
        finish_prewarm_run(
            prewarm_id=run.id,
            promoted_count=promoted_count,
            pools_created=pools_created,
        )
        promoted_question_ids = [str(qid) for qid in (result.get("promoted_question_ids") or [])]
        return AdminPrewarmPoolsResponse(
            status=str(result.get("status") or "success"),
            prewarm_id=run.id,
            job_id=run.id,
            questions_promoted=promoted_count,
            pools_created=pools_created,
            promoted_question_ids=promoted_question_ids,
        )
    except Exception as e:
        fail_prewarm_run(prewarm_id=run.id, error_message=str(e))
        raise


@router.get("/prewarm/{prewarm_id}/questions", response_model=AdminPrewarmQuestionsResponse)
def admin_get_prewarm_questions(request: Request, prewarm_id: str):
    questions = get_prewarm_promoted_questions(prewarm_id=prewarm_id)
    if not questions:
        # Distinguish empty run from not found is intentionally not exposed;
        # caller should treat missing data as no promoted questions.
        return AdminPrewarmQuestionsResponse(prewarm_id=prewarm_id, questions=[])
    return AdminPrewarmQuestionsResponse(prewarm_id=prewarm_id, questions=questions)


@router.post("/invalidate-pools")
def admin_invalidate_pools(request: Request):
    """
    Internal-only pool invalidation for syllabus updates.
    """
    syllabus = request.headers.get("X-Syllabus", "CBSE")
    syllabus_version = request.headers.get("X-Syllabus-Version", "2024")
    result = invalidate_pools_for_syllabus(
        syllabus=syllabus,
        syllabus_version=syllabus_version,
    )
    return {
        "status": "ok",
        "syllabus": syllabus,
        "syllabus_version": syllabus_version,
        **result,
    }


@router.post("/reindex", response_model=AdminReindexResponse)
def admin_reindex_documents(request: Request, req: AdminReindexRequest):
    """
    Admin endpoint to re-index existing documents.
    Creates a single index job for the provided document IDs.
    """
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")

    with get_session() as session:
        docs = session.exec(select(Document).where(Document.id.in_(req.document_ids))).all()
        if not docs or len(docs) != len(req.document_ids):
            raise HTTPException(status_code=404, detail="some_documents_not_found")

        corpus_ids = {doc.corpus_id for doc in docs}
        if len(corpus_ids) != 1:
            raise HTTPException(status_code=400, detail="documents_multiple_corpus_not_supported")
        corpus_id = next(iter(corpus_ids))

        corpus = session.get(Corpus, corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")

        job = GenerationJob(
            kind=JobKind.index,
            status=JobStatus.queued,
            tenant_id=tenant_id,
            corpus_id=corpus_id,
            request_json=json.dumps({"document_ids": req.document_ids}),
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        log.info(
            "admin_reindex_job_created",
            job_id=job.id,
            corpus_id=corpus_id,
            documents=len(req.document_ids),
            tenant_id=tenant_id,
        )

        return AdminReindexResponse(
            job_id=job.id,
            status="queued",
            documents=len(req.document_ids),
            message=f"Re-index job queued for {len(req.document_ids)} document(s).",
        )

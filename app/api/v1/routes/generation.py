from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.api.v1.schemas.generation import GenerateJobResponse, GenerateRequest, GenerateResponse
from app.api.v1.schemas.pools import PoolRequest, PoolResponse
from app.db.models import Corpus, GenerationJob, JobKind, JobStatus, QuestionPoolStatus
from app.db.session import get_session
from app.generation.sync import generate_questions_sync
from app.observability.logging import log
from app.pools.service import (
    get_or_create_pool,
    get_pool_status,
    get_questions_from_pool,
)


router = APIRouter()


@router.post("", response_model=GenerateResponse)
def generate_sync(request: Request, req: GenerateRequest):
    """
    DEPRECATED: Synchronous generation (kept for backward compatibility).
    This endpoint still calls LLMs synchronously and should be avoided.
    Use /pools endpoint instead for fast, cache-based question serving.
    """
    if req.blueprint.count > 10:
        raise HTTPException(status_code=400, detail="count_too_large_for_sync_use_jobs")
    with get_session() as session:
        corpus = session.get(Corpus, req.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != req.tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")

    log.warning("deprecated_sync_endpoint_used", tenant_id=req.tenant_id, corpus_id=req.corpus_id)
    return generate_questions_sync(req)


@router.post("/jobs", response_model=GenerateJobResponse)
def generate_async(request: Request, req: GenerateRequest):
    with get_session() as session:
        corpus = session.get(Corpus, req.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != req.tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")

        job = GenerationJob(
            kind=JobKind.generate,
            status=JobStatus.queued,
            tenant_id=req.tenant_id,
            corpus_id=req.corpus_id,
            request_json=req.model_dump_json(),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        return GenerateJobResponse(job_id=job.id, status=job.status.value)


@router.get("/jobs/{job_id}/result", response_model=GenerateResponse)
def get_generation_result(request: Request, job_id: str):
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        if job.kind.value != "GENERATE":
            raise HTTPException(status_code=400, detail="not_a_generation_job")
        if job.status.value != "SUCCEEDED":
            raise HTTPException(status_code=409, detail=f"job_not_succeeded:{job.status.value}")
        assert job.result_json is not None
        return GenerateResponse.model_validate(json.loads(job.result_json))


@router.post("/pools", response_model=PoolResponse)
def serve_from_pool(request: Request, req: PoolRequest):
    """
    FAST PATH: Serve questions from pre-generated pools (NO LLM calls).
    
    This is the primary endpoint for question serving.
    - Checks Redis for available questions
    - Returns immediately if pool is ready
    - Returns 202 if pool is generating
    - Triggers background generation if pool doesn't exist
    """
    # Validate corpus
    with get_session() as session:
        corpus = session.get(Corpus, req.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != req.tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")
    
    # Check each difficulty pool
    all_questions = []
    pool_info = {}
    overall_status = "ready"
    
    for difficulty, count in req.difficulty_mix.items():
        if count <= 0:
            continue
        
        # Get or create pool
        pool = get_or_create_pool(
            syllabus=req.syllabus,
            class_level=req.class_level,
            subject=req.subject,
            difficulty=difficulty,
            syllabus_version=req.syllabus_version,
        )
        pool_info[difficulty] = pool.id
        
        pool_status = get_pool_status(
            syllabus=req.syllabus,
            class_level=req.class_level,
            subject=req.subject,
            difficulty=difficulty,
            syllabus_version=req.syllabus_version,
        )
        
        if pool_status in (QuestionPoolStatus.ready, QuestionPoolStatus.ready_partial):
            # Fast path: fetch from Redis
            questions = get_questions_from_pool(
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
                count=count,
            )
            
            if len(questions) >= count:
                # Convert PoolQuestion to GeneratedQuestionOut format
                for pq in questions[:count]:
                    q_data = json.loads(pq.question_json)
                    all_questions.append(q_data)
            else:
                # Partial/insufficient: return what we have and trigger backfill
                if questions:
                    for pq in questions:
                        q_data = json.loads(pq.question_json)
                        all_questions.append(q_data)
                    overall_status = "ready_partial"
                else:
                    overall_status = "generating"
                _trigger_pool_generation(
                    corpus_id=req.corpus_id,
                    syllabus=req.syllabus,
                    class_level=req.class_level,
                    subject=req.subject,
                    difficulty=difficulty,
                    syllabus_version=req.syllabus_version,
                )
        
        elif pool_status == QuestionPoolStatus.generating:
            overall_status = "generating"
        
        elif pool_status == QuestionPoolStatus.pending or pool_status is None:
            # Trigger generation
            overall_status = "generating"
            _trigger_pool_generation(
                corpus_id=req.corpus_id,
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
            )
        
        elif pool_status == QuestionPoolStatus.failed:
            # Retry failed pool
            overall_status = "generating"
            _trigger_pool_generation(
                corpus_id=req.corpus_id,
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
            )
    
    # Return response
    if overall_status in ("ready", "ready_partial") and all_questions:
        return PoolResponse(
            questions=all_questions,
            status=overall_status,
            pool_info=pool_info,
        )
    else:
        # Return 202 Accepted - pool is generating
        return PoolResponse(
            questions=None,
            status="generating",
            pool_info=pool_info,
        )


def _trigger_pool_generation(
    corpus_id: str,
    syllabus: str,
    class_level: int,
    subject: str,
    difficulty: str,
    syllabus_version: str,
) -> None:
    """Trigger background job to generate question pool."""
    with get_session() as session:
        # Check if job already exists (simplified check)
        existing_jobs = session.exec(
            select(GenerationJob).where(
                GenerationJob.kind == JobKind.generate_question_pool,
                GenerationJob.corpus_id == corpus_id,
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
        
        # Create new job
        job_data = {
            "corpus_id": corpus_id,
            "syllabus": syllabus,
            "class_level": class_level,
            "subject": subject,
            "difficulty": difficulty,
            "syllabus_version": syllabus_version,
        }
        
        job = GenerationJob(
            kind=JobKind.generate_question_pool,
            status=JobStatus.queued,
            tenant_id="system",  # System-level job
            corpus_id=corpus_id,
            request_json=json.dumps(job_data),
        )
        session.add(job)
        session.commit()
        log.info("pool_generation_job_created", job_id=job.id, difficulty=difficulty)



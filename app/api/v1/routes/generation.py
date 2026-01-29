from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.schemas.generation import GenerateJobResponse, GenerateRequest, GenerateResponse
from app.api.v1.schemas.pools import PoolRequest, PoolResponse
from app.db.models import Corpus, GenerationJob, JobKind, JobStatus, QuestionPoolStatus
from app.db.session import get_session
from app.observability.logging import log
from app.pools.service import (
    get_or_create_pool,
    get_pool_status,
    get_questions_from_difficulty_pool,
    list_pool_chapters,
)


router = APIRouter()


@router.post("", response_model=GenerateResponse)
def generate_sync(request: Request, req: GenerateRequest):
    """
    DEPRECATED: Synchronous generation (kept for backward compatibility).
    This endpoint still calls LLMs synchronously and should be avoided.
    Use /pools endpoint instead for fast, cache-based question serving.
    """
    raise HTTPException(status_code=403, detail="sync_generation_disabled_use_pools")


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
    - Returns 202 if pool is not ready
    """
    # Validate corpus
    with get_session() as session:
        corpus = session.get(Corpus, req.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != req.tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")
    
    chapters = list_pool_chapters(req.corpus_id, req.class_level, req.subject)

    # Check each difficulty pool
    all_questions = []
    pool_info = {}
    pool_info_by_chapter: dict[str, dict[str, str]] = {}
    overall_status = "ready"
    
    for difficulty, count in req.difficulty_mix.items():
        if count <= 0:
            continue
        log.info(
            "pool_request_difficulty_mix",
            class_level=req.class_level,
            subject=req.subject,
            difficulty=difficulty,
            target_count=count,
            corpus_id=req.corpus_id,
        )
        
        pool_info_by_chapter[difficulty] = {}
        pool_statuses: list[str] = []
        count_remaining = count

        for chapter in chapters:
            pool = get_or_create_pool(
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                chapter=chapter,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
            )
            pool_info_by_chapter[difficulty][chapter] = pool.id

            pool_status = get_pool_status(
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                chapter=chapter,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
            )
            pool_statuses.append(pool_status.value if pool_status else "pending")

        if any(status in (QuestionPoolStatus.ready.value, QuestionPoolStatus.ready_partial.value) for status in pool_statuses):
            questions = get_questions_from_difficulty_pool(
                syllabus=req.syllabus,
                class_level=req.class_level,
                subject=req.subject,
                difficulty=difficulty,
                syllabus_version=req.syllabus_version,
                count=count_remaining,
            )
            if questions:
                for pq in questions:
                    q_data = json.loads(pq.question_json)
                    all_questions.append(q_data)
                count_remaining -= len(questions)

        if count_remaining <= 0:
            pool_info[difficulty] = next(iter(pool_info_by_chapter[difficulty].values()))
            continue

        if any(status == QuestionPoolStatus.generating.value for status in pool_statuses):
            overall_status = "generating"
        elif any(status == QuestionPoolStatus.pending.value for status in pool_statuses):
            overall_status = "generating"
        elif any(status == QuestionPoolStatus.failed.value for status in pool_statuses):
            overall_status = "generating"
        elif any(status == QuestionPoolStatus.ready_partial.value for status in pool_statuses):
            overall_status = "ready_partial"

        pool_info[difficulty] = next(iter(pool_info_by_chapter[difficulty].values()))
    
    # Return response
    if overall_status in ("ready", "ready_partial") and all_questions:
        return PoolResponse(
            questions=all_questions,
            status=overall_status,
            pool_info=pool_info,
            pool_info_by_chapter=pool_info_by_chapter,
        )
    else:
        # Return 202 Accepted - pool is not ready (no generation triggered here)
        return PoolResponse(
            questions=None,
            status="generating",
            pool_info=pool_info,
            pool_info_by_chapter=pool_info_by_chapter,
        )



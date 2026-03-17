from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.schemas.jobs import JobStatusResponse
from app.db.models import GenerationJob, JobStatus
from app.db.session import get_session_ctx


router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(request: Request, job_id: str):
    import json
    
    async with get_session_ctx() as session:
        job = await session.get(GenerationJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        
        err = None
        if job.status in (JobStatus.failed_retryable, JobStatus.failed_terminal):
            err = {
                "code": job.error_code or "JOB_FAILED",
                "message": job.error_message or job.last_error or "job_failed",
                "retryable": job.status == JobStatus.failed_retryable,
            }
        
        result = None
        if job.result_json:
            try:
                result = json.loads(job.result_json)
            except (json.JSONDecodeError, TypeError):
                result = {"raw": job.result_json}
        
        return JobStatusResponse(
            job_id=job.id,
            kind=job.kind.value,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=err,
            result=result,
        )

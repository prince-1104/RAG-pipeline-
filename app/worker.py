from __future__ import annotations

import json
import time
from datetime import datetime

from sqlmodel import select

from app.db.models import GenerationJob, JobKind, JobStatus
from app.db.session import get_session, init_db
from app.indexing.index_job import run_index_job
from app.observability.logging import configure_logging, log
from app.settings import get_settings
from app.tasks.admin_generate_job import run_admin_generate_job
from app.tasks.generate_job import run_generate_job
from app.tasks.pool_generation import generate_question_pool


POLL_INTERVAL_S = 1.0


def _claim_next_job() -> GenerationJob | None:
    """
    Simple DB-backed queue for this repo (no Kafka).
    In prod we can harden with SKIP LOCKED + multiple workers.
    """
    with get_session() as session:
        job = session.exec(
            select(GenerationJob)
            .where(GenerationJob.status == JobStatus.queued)
            .order_by(GenerationJob.created_at.asc())
            .limit(1)
        ).first()
        if not job:
            return None
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def _finish_job_success(job_id: str, result: dict | None = None) -> None:
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        assert job is not None
        job.status = JobStatus.succeeded
        job.finished_at = datetime.utcnow()
        if result is not None:
            job.result_json = json.dumps(result)
        session.add(job)
        session.commit()


def _finish_job_failure(job_id: str, code: str, message: str) -> None:
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        assert job is not None
        job.status = JobStatus.failed
        job.finished_at = datetime.utcnow()
        job.error_code = code
        job.error_message = message
        session.add(job)
        session.commit()


def run_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    log.info("worker_started", env=settings.env)

    while True:
        job = _claim_next_job()
        if not job:
            time.sleep(POLL_INTERVAL_S)
            continue

        log.info("job_claimed", job_id=job.id, kind=job.kind.value, tenant_id=job.tenant_id, corpus_id=job.corpus_id)
        try:
            if job.kind == JobKind.index:
                run_index_job(job)
                # Preserve progress data that run_index_job wrote to result_json
                with get_session() as session:
                    updated_job = session.get(GenerationJob, job.id)
                    if updated_job and updated_job.result_json:
                        try:
                            progress_data = json.loads(updated_job.result_json)
                            # Ensure it has ok flag
                            progress_data["ok"] = True
                            _finish_job_success(job.id, result=progress_data)
                        except (json.JSONDecodeError, TypeError):
                            # Fallback if result_json is invalid
                            _finish_job_success(job.id, result={"ok": True})
                    else:
                        _finish_job_success(job.id, result={"ok": True})
            elif job.kind == JobKind.generate:
                result = run_generate_job(job)
                _finish_job_success(job.id, result=result)
            elif job.kind == JobKind.generate_question_pool:
                # Background question pool generation
                job_data = json.loads(job.request_json)
                generate_question_pool(
                    corpus_id=job_data["corpus_id"],
                    syllabus=job_data["syllabus"],
                    class_level=job_data["class_level"],
                    subject=job_data["subject"],
                    difficulty=job_data["difficulty"],
                    syllabus_version=job_data.get("syllabus_version", "2024"),
                    target_count=200,  # Generate 200 questions per pool
                )
                _finish_job_success(job.id, result={"ok": True, "pool_generated": True})
            elif job.kind == JobKind.refresh_question_pool:
                # Refresh existing pool (same as generate but clears old questions first)
                job_data = json.loads(job.request_json)
                # TODO: Clear old questions from pool before regenerating
                generate_question_pool(
                    corpus_id=job_data["corpus_id"],
                    syllabus=job_data["syllabus"],
                    class_level=job_data["class_level"],
                    subject=job_data["subject"],
                    difficulty=job_data["difficulty"],
                    syllabus_version=job_data.get("syllabus_version", "2024"),
                    target_count=200,
                )
                _finish_job_success(job.id, result={"ok": True, "pool_refreshed": True})
            elif job.kind == JobKind.admin_generate_questions:
                # Admin-triggered question generation (ONLY place LLM calls are allowed)
                result = run_admin_generate_job(job)
                _finish_job_success(job.id, result=result)
            else:
                raise RuntimeError(f"unsupported_job_kind:{job.kind.value}")
        except Exception as e:  # noqa: BLE001
            log.exception("job_failed", job_id=job.id, kind=job.kind.value)
            _finish_job_failure(job.id, code="JOB_FAILED", message=str(e))


if __name__ == "__main__":
    run_forever()



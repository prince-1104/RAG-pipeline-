from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timedelta

from sqlmodel import select
from sqlalchemy import update, and_, or_, func

from app.db.models import Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session, init_db
from app.indexing.index_job import run_index_job
from app.indexing.qdrant_index import ensure_qdrant_payload_indexes
from app.observability.logging import configure_logging, get_process_rss_mb, log
from app.settings import get_settings
from app.tasks.admin_generate_job import run_admin_generate_job
from app.tasks.generate_job import RetryableError, TerminalError, run_generate_job
from app.tasks.pool_generation import generate_question_pool


POLL_INTERVAL_S = 1.0

# ANSI color codes for terminal highlights
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _banner(color: str, label: str, detail: str) -> None:
    """Print a colored banner line to stdout for quick visual scanning."""
    import sys
    print(f"{color}{_BOLD}{'='*60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"  {detail}", file=sys.stderr)
    print(f"{'='*60}{_RESET}", file=sys.stderr, flush=True)


def _get_job_counts() -> tuple[int, int]:
    """Get counts of completed jobs and total active jobs (completed + pending)."""
    with get_session() as session:
        completed_count = session.exec(
            select(func.count(GenerationJob.id))
            .where(
                GenerationJob.status.in_([
                    JobStatus.completed,
                    JobStatus.completed_partial,
                ])
            )
        ).one() or 0
        
        pending_count = session.exec(
            select(func.count(GenerationJob.id))
            .where(
                GenerationJob.status.in_([
                    JobStatus.queued,
                    JobStatus.claimed,
                    JobStatus.running,
                    JobStatus.failed_retryable,
                ])
            )
        ).one() or 0
        
        total_active = completed_count + pending_count
        return completed_count, total_active


def _get_total_statistics() -> dict:
    """Get total statistics from completed jobs."""
    with get_session() as session:
        # Get all completed jobs
        completed_jobs = session.exec(
            select(GenerationJob)
            .where(
                GenerationJob.status.in_([
                    JobStatus.completed,
                    JobStatus.completed_partial,
                ])
            )
        ).all()
        
        total_questions = 0
        total_chunks = 0
        total_time = 0.0
        
        for job in completed_jobs:
            if job.result_json:
                try:
                    result = json.loads(job.result_json)
                    # Extract questions count from generation jobs
                    if job.kind == JobKind.generate:
                        meta = result.get("meta", {})
                        gen_count = meta.get("generated_count", 0)
                        if isinstance(gen_count, (int, float)):
                            total_questions += int(gen_count)
                    # Extract chunks from index jobs
                    elif job.kind == JobKind.index:
                        chunks = result.get("chunks_indexed") or result.get("chunks_embedded") or result.get("chunks_created", 0)
                        if isinstance(chunks, (int, float)):
                            total_chunks += int(chunks)
                    # Calculate job time
                    if job.started_at and job.finished_at:
                        job_time = (job.finished_at - job.started_at).total_seconds()
                        total_time += job_time
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return {
            "total_questions": total_questions,
            "total_chunks": total_chunks,
            "total_time": total_time,
        }


def _check_and_display_all_jobs_done(worker_stats: dict | None = None) -> None:
    """
    Check if there are any pending jobs and display 'ALL JOBS DONE' with summary if queue is empty.
    Uses session-scoped stats: only shows metrics for work done since last reset.
    Resets worker_stats after display so next batch shows only its own stats.
    """
    with get_session() as session:
        pending_jobs = session.exec(
            select(GenerationJob)
            .where(
                GenerationJob.status.in_([
                    JobStatus.queued,
                    JobStatus.claimed,
                    JobStatus.running,
                    JobStatus.failed_retryable,
                ])
            )
        ).first()
        if pending_jobs is None:
            # No pending jobs - all done! Get statistics for this batch only
            if worker_stats:
                stats = dict(worker_stats)  # Snapshot before reset
                # Reset so next batch starts fresh
                worker_stats["total_questions"] = 0
                worker_stats["total_chunks"] = 0
                worker_stats["total_time"] = 0.0
            else:
                stats = _get_total_statistics()
            
            total_time_str = f"{stats.get('total_time', 0):.1f}s" if stats.get('total_time', 0) > 0 else "N/A"
            
            summary_parts = []
            if stats.get('total_questions', 0) > 0:
                summary_parts.append(f"questions={stats['total_questions']}")
            if stats.get('total_chunks', 0) > 0:
                summary_parts.append(f"chunks={stats['total_chunks']}")
            summary_parts.append(f"total_time={total_time_str}")
            
            summary = " | ".join(summary_parts) if summary_parts else "All tasks completed"
            _banner(_CYAN, "🎉 ALL JOBS DONE 🎉", summary)


def _claim_next_job() -> GenerationJob | None:
    """DB-backed queue with SKIP LOCKED for safe multi-worker claims."""
    with get_session() as session:
        now = datetime.utcnow()
        settings = get_settings()
        stale_cutoff = now - timedelta(seconds=int(settings.job_stale_timeout_s))
        # Requeue stale claimed/running jobs so they can be retried after crashes.
        session.exec(
            update(GenerationJob)
            .where(GenerationJob.status.in_([JobStatus.claimed, JobStatus.running]))
            .where(
                or_(
                    and_(
                        GenerationJob.started_at.is_not(None),
                        GenerationJob.started_at < stale_cutoff,
                    ),
                    and_(
                        GenerationJob.claimed_at.is_not(None),
                        GenerationJob.claimed_at < stale_cutoff,
                    ),
                )
            )
            .values(status=JobStatus.queued, started_at=None, claimed_at=None, worker_id=None)
        )
        session.commit()
        # Atomic claim: select + lock one eligible job.
        claimable = (
            select(GenerationJob)
            .where(
                GenerationJob.status.in_([JobStatus.queued, JobStatus.failed_retryable]),
                or_(
                    GenerationJob.next_retry_at.is_(None),
                    GenerationJob.next_retry_at <= now,
                ),
            )
            .order_by(GenerationJob.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = session.exec(claimable).first()
        if not job:
            return None
        worker_id = settings.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        job.status = JobStatus.claimed
        job.claimed_at = now
        job.worker_id = worker_id
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def _mark_job_running(job_id: str) -> None:
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        assert job is not None
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        session.add(job)
        session.commit()


def _finish_job_completed(job_id: str, *, status: JobStatus, result: dict | None = None) -> None:
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        assert job is not None
        job.status = status
        job.finished_at = datetime.utcnow()
        if result is not None:
            job.result_json = json.dumps(result)
        session.add(job)
        session.commit()


def _mark_job_failed(
    job_id: str,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        assert job is not None
        settings = get_settings()
        max_retries = int(settings.job_max_retries)
        base_delay = int(settings.job_retry_base_delay_s)
        max_delay = int(settings.job_retry_max_delay_s)
        next_retry_at = None
        if retryable and int(job.retry_count or 0) < max_retries:
            job.retry_count = int(job.retry_count or 0) + 1
            delay_s = min(max_delay, base_delay * (2 ** max(0, job.retry_count - 1)))
            next_retry_at = datetime.utcnow() + timedelta(seconds=delay_s)
            job.status = JobStatus.failed_retryable
            job.next_retry_at = next_retry_at
        else:
            job.status = JobStatus.failed_terminal
        job.finished_at = datetime.utcnow()
        job.error_code = code
        job.error_message = message
        job.last_error = message
        session.add(job)
        session.commit()


def run_forever() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    ensure_qdrant_payload_indexes()
    log.info(
        "worker_started",
        env=settings.env,
        embedding_hard_cap_tokens=int(settings.embedding_hard_cap_tokens),
    )
    
    # Track worker-level statistics
    worker_start_time = time.perf_counter()
    jobs_completed_count = 0
    worker_stats = {
        "total_questions": 0,
        "total_chunks": 0,
        "total_time": 0.0,
    }

    while True:
        job = _claim_next_job()
        if not job:
            time.sleep(POLL_INTERVAL_S)
            continue

        rss_mb = get_process_rss_mb()
        
        # Get job progress info
        completed_count, total_active = _get_job_counts()
        job_progress = f"job({completed_count + 1}/{total_active})" if total_active > 0 else "job(?)"
        
        log.info(
            "job_claimed",
            job_id=job.id,
            kind=job.kind.value,
            tenant_id=job.tenant_id,
            corpus_id=job.corpus_id,
            rss_mb=round(rss_mb, 2) if rss_mb is not None else None,
        )
        try:
            job_start = time.perf_counter()

            if job.kind == JobKind.index:
                _mark_job_running(job.id)

                def _on_doc_complete(doc_idx: int, doc_total: int, doc: Document, chunks: int) -> None:
                    doc_progress = f"doc({doc_idx}/{doc_total})"
                    filename = getattr(doc, "filename", None) or str(doc.id)[:8]
                    _banner(_GREEN, "DOC INDEXED", f"{doc_progress}  document_id={doc.id[:8]}...  filename={filename}  chunks={chunks}")

                run_index_job(job, on_document_complete=_on_doc_complete)
                # Preserve progress data that run_index_job wrote to result_json
                with get_session() as session:
                    updated_job = session.get(GenerationJob, job.id)
                    if updated_job and updated_job.result_json:
                        try:
                            progress_data = json.loads(updated_job.result_json)
                            progress_data["ok"] = True
                            _finish_job_completed(job.id, status=JobStatus.completed, result=progress_data)
                        except (json.JSONDecodeError, TypeError):
                            _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True})
                    else:
                        _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True})
                elapsed = round(time.perf_counter() - job_start, 1)
                jobs_completed_count += 1
                worker_stats["total_time"] += elapsed
                
                # Extract chunks and doc progress from result for banner
                doc_progress = ""
                with get_session() as session:
                    updated_job = session.get(GenerationJob, job.id)
                    if updated_job and updated_job.result_json:
                        try:
                            result = json.loads(updated_job.result_json)
                            chunks_val = result.get("chunks_indexed") or result.get("chunks_embedded") or result.get("chunks_created", 0)
                            if isinstance(chunks_val, (int, float)):
                                worker_stats["total_chunks"] += int(chunks_val)
                            doc_total = result.get("documents_total", 0)
                            doc_done = result.get("documents_processed", 0)
                            if doc_total > 0:
                                doc_progress = f"  doc({doc_done}/{doc_total})  total_chunks={chunks_val}"
                        except (json.JSONDecodeError, TypeError):
                            pass
                completed_count, total_active = _get_job_counts()
                job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count})"
                _banner(_GREEN, "INDEX COMPLETE", f"{job_progress}  job_id={job.id}  corpus={job.corpus_id}{doc_progress}  time={elapsed}s")
                _check_and_display_all_jobs_done(worker_stats)

            elif job.kind == JobKind.generate:
                _mark_job_running(job.id)
                result = run_generate_job(job)
                elapsed = round(time.perf_counter() - job_start, 1)
                if result["final_status"] in (JobStatus.completed, JobStatus.completed_partial):
                    _finish_job_completed(job.id, status=result["final_status"], result=result["payload"])
                    meta = result.get("payload", {}).get("meta", {})
                    gen_count = meta.get("generated_count", "?")
                    target = meta.get("target_count", "?")
                    jobs_completed_count += 1
                    worker_stats["total_time"] += elapsed
                    
                    # Track questions generated
                    if isinstance(gen_count, (int, float)):
                        worker_stats["total_questions"] += int(gen_count)
                    
                    completed_count, total_active = _get_job_counts()
                    job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count})"
                    _banner(_GREEN, "GENERATION COMPLETE", f"{job_progress}  job_id={job.id}  questions={gen_count}/{target}  status={result['final_status'].value}  time={elapsed}s")
                    _check_and_display_all_jobs_done(worker_stats)
                else:
                    _mark_job_failed(
                        job.id,
                        code=result.get("error_code") or "JOB_FAILED",
                        message=result.get("error_message") or "job_failed",
                        retryable=result["final_status"] == JobStatus.failed_retryable,
                    )
                    _banner(_RED, "GENERATION FAILED", f"job={job.id}  error={result.get('error_code','?')}  time={elapsed}s")

            elif job.kind == JobKind.generate_question_pool:
                _mark_job_running(job.id)
                job_data = json.loads(job.request_json)
                generate_question_pool(
                    corpus_id=job_data["corpus_id"],
                    syllabus=job_data["syllabus"],
                    class_level=job_data["class_level"],
                    subject=job_data["subject"],
                    chapter=job_data.get("chapter", "Unknown Chapter"),
                    difficulty=job_data["difficulty"],
                    syllabus_version=job_data.get("syllabus_version", "2024"),
                    target_count=int(job_data.get("target_count") or 200),
                )
                _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True, "pool_generated": True})
                elapsed = round(time.perf_counter() - job_start, 1)
                jobs_completed_count += 1
                worker_stats["total_time"] += elapsed
                completed_count, total_active = _get_job_counts()
                job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count})"
                _banner(_GREEN, "POOL GENERATION COMPLETE", f"{job_progress}  job_id={job.id}  chapter={job_data.get('chapter','?')}  time={elapsed}s")
                _check_and_display_all_jobs_done(worker_stats)

            elif job.kind == JobKind.refresh_question_pool:
                _mark_job_running(job.id)
                job_data = json.loads(job.request_json)
                generate_question_pool(
                    corpus_id=job_data["corpus_id"],
                    syllabus=job_data["syllabus"],
                    class_level=job_data["class_level"],
                    subject=job_data["subject"],
                    chapter=job_data.get("chapter", "Unknown Chapter"),
                    difficulty=job_data["difficulty"],
                    syllabus_version=job_data.get("syllabus_version", "2024"),
                    target_count=int(job_data.get("target_count") or 200),
                )
                _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True, "pool_refreshed": True})
                elapsed = round(time.perf_counter() - job_start, 1)
                jobs_completed_count += 1
                worker_stats["total_time"] += elapsed
                completed_count, total_active = _get_job_counts()
                job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count})"
                _banner(_GREEN, "POOL REFRESH COMPLETE", f"{job_progress}  job_id={job.id}  time={elapsed}s")
                _check_and_display_all_jobs_done(worker_stats)

            elif job.kind == JobKind.admin_generate_questions:
                _mark_job_running(job.id)
                result = run_admin_generate_job(job)
                _finish_job_completed(job.id, status=JobStatus.completed, result=result)
                elapsed = round(time.perf_counter() - job_start, 1)
                jobs_completed_count += 1
                worker_stats["total_time"] += elapsed
                
                # Extract questions from admin generation result
                if isinstance(result, dict):
                    admin_meta = result.get("meta", {})
                    admin_gen_count = admin_meta.get("generated_count", 0)
                    if isinstance(admin_gen_count, (int, float)):
                        worker_stats["total_questions"] += int(admin_gen_count)
                
                completed_count, total_active = _get_job_counts()
                job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count})"
                _banner(_GREEN, "ADMIN GENERATION COMPLETE", f"{job_progress}  job_id={job.id}  time={elapsed}s")
                _check_and_display_all_jobs_done(worker_stats)

            else:
                raise RuntimeError(f"unsupported_job_kind:{job.kind.value}")
        except RetryableError as e:
            elapsed = round(time.perf_counter() - job_start, 1)
            log.exception("job_failed_retryable", job_id=job.id, kind=job.kind.value)
            _mark_job_failed(job.id, code="JOB_RETRYABLE", message=str(e), retryable=True)
            _banner(_YELLOW, "JOB FAILED (retryable)", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}  time={elapsed}s")
        except TerminalError as e:
            elapsed = round(time.perf_counter() - job_start, 1)
            log.exception("job_failed_terminal", job_id=job.id, kind=job.kind.value)
            _mark_job_failed(job.id, code="JOB_TERMINAL", message=str(e), retryable=False)
            _banner(_RED, "JOB FAILED (terminal)", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}  time={elapsed}s")
        except Exception as e:  # noqa: BLE001
            elapsed = round(time.perf_counter() - job_start, 1)
            log.exception("job_failed", job_id=job.id, kind=job.kind.value)
            _mark_job_failed(job.id, code="JOB_FAILED", message=str(e), retryable=True)
            _banner(_RED, "JOB FAILED", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}  time={elapsed}s")


if __name__ == "__main__":
    run_forever()



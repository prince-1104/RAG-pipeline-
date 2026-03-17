from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
from datetime import datetime, timedelta

from sqlalchemy import select, update, and_, or_, func

from app.db.models import Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session_ctx, init_db
from app.indexing.index_job import run_index_job
from app.indexing.qdrant_index import ensure_qdrant_payload_indexes
from app.observability.logging import configure_logging, get_process_rss_mb, log
from app.settings import Settings, get_settings
from app.tasks.admin_generate_job import run_admin_generate_job
from app.tasks.generate_job import RetryableError, TerminalError, run_generate_job
from app.tasks.pool_generation import generate_question_pool


_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_shutdown_event = asyncio.Event()


def _banner(color: str, label: str, detail: str) -> None:
    """Print a banner; detail can contain newlines for multiple lines."""
    print(f"{color}{_BOLD}{'='*60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    for line in detail.strip().split("\n"):
        print(f"  {line}", file=sys.stderr)
    print(f"{'='*60}{_RESET}", file=sys.stderr, flush=True)


def _handle_signal() -> None:
    _shutdown_event.set()


async def _get_job_counts() -> tuple[int, int]:
    async with get_session_ctx() as session:
        completed_result = await session.execute(
            select(func.count(GenerationJob.id))
            .where(
                GenerationJob.status.in_([
                    JobStatus.completed,
                    JobStatus.completed_partial,
                ])
            )
        )
        completed_count = completed_result.scalars().one() or 0

        pending_result = await session.execute(
            select(func.count(GenerationJob.id))
            .where(
                GenerationJob.status.in_([
                    JobStatus.queued,
                    JobStatus.claimed,
                    JobStatus.running,
                    JobStatus.failed_retryable,
                ])
            )
        )
        pending_count = pending_result.scalars().one() or 0

        total_active = completed_count + pending_count
        return completed_count, total_active


async def _get_total_statistics() -> dict:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(GenerationJob)
            .where(
                GenerationJob.status.in_([
                    JobStatus.completed,
                    JobStatus.completed_partial,
                ])
            )
        )
        completed_jobs = result.scalars().all()

        total_questions = 0
        total_chunks = 0
        total_time = 0.0

        for job in completed_jobs:
            if job.result_json:
                try:
                    r = json.loads(job.result_json)
                    if job.kind == JobKind.generate:
                        meta = r.get("meta", {})
                        gen_count = meta.get("generated_count", 0)
                        if isinstance(gen_count, (int, float)):
                            total_questions += int(gen_count)
                    elif job.kind == JobKind.index:
                        chunks = r.get("chunks_indexed") or r.get("chunks_embedded") or r.get("chunks_created", 0)
                        if isinstance(chunks, (int, float)):
                            total_chunks += int(chunks)
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


async def _check_and_display_all_jobs_done(worker_stats: dict | None = None) -> None:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(GenerationJob)
            .where(
                GenerationJob.status.in_([
                    JobStatus.queued,
                    JobStatus.claimed,
                    JobStatus.running,
                    JobStatus.failed_retryable,
                ])
            )
        )
        pending_jobs = result.scalars().first()
        if pending_jobs is None:
            completed_count, total_active = await _get_job_counts()
            if worker_stats:
                stats = dict(worker_stats)
                worker_stats["total_questions"] = 0
                worker_stats["total_chunks"] = 0
                worker_stats["total_time"] = 0.0
            else:
                stats = await _get_total_statistics()

            total_time_str = f"{stats.get('total_time', 0):.1f}s" if stats.get('total_time', 0) > 0 else "N/A"

            summary_parts = []
            summary_parts.append(f"jobs={completed_count}/{total_active}  status: all jobs finished")
            if stats.get('total_questions', 0) > 0:
                summary_parts.append(f"questions={stats['total_questions']}")
            if stats.get('total_chunks', 0) > 0:
                summary_parts.append(f"chunks={stats['total_chunks']}")
            summary_parts.append(f"total_time={total_time_str}")

            summary = " | ".join(summary_parts)
            _banner(_CYAN, "ALL JOBS DONE", summary)


async def _claim_next_job(job_kinds: list[str] | None = None) -> GenerationJob | None:
    async with get_session_ctx() as session:
        now = datetime.utcnow()
        settings = get_settings()
        stale_cutoff = now - timedelta(seconds=int(settings.job_stale_timeout_s))

        await session.execute(
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
        await session.commit()

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
        if job_kinds:
            claimable = claimable.where(GenerationJob.kind.in_(job_kinds))

        result = await session.execute(claimable)
        job = result.scalars().first()
        if not job:
            return None

        max_retries = getattr(settings, "generation_job_max_retries", None) or int(settings.job_max_retries)
        if (job.retry_count or 0) >= max_retries:
            log.error(
                "job_max_retries_exceeded",
                job_id=job.id,
                retry_count=job.retry_count or 0,
                max_retries=max_retries,
            )
            job.status = JobStatus.failed_terminal
            job.error_message = f"Exceeded max retries ({max_retries})"
            job.finished_at = datetime.utcnow()
            session.add(job)
            await session.commit()
            return None

        worker_id = settings.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        job.status = JobStatus.claimed
        job.claimed_at = now
        job.worker_id = worker_id
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job


async def _mark_job_running(job_id: str) -> None:
    async with get_session_ctx() as session:
        job = await session.get(GenerationJob, job_id)
        assert job is not None
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        session.add(job)
        await session.commit()


async def _finish_job_completed(job_id: str, *, status: JobStatus, result: dict | None = None) -> None:
    async with get_session_ctx() as session:
        job = await session.get(GenerationJob, job_id)
        assert job is not None
        job.status = status
        job.finished_at = datetime.utcnow()
        if result is not None:
            job.result_json = json.dumps(result)
        session.add(job)
        await session.commit()


async def _mark_job_failed(
    job_id: str,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    async with get_session_ctx() as session:
        job = await session.get(GenerationJob, job_id)
        assert job is not None
        settings = get_settings()
        max_retries = int(settings.job_max_retries)
        base_delay = int(settings.job_retry_base_delay_s)
        max_delay = int(settings.job_retry_max_delay_s)
        if retryable and int(job.retry_count or 0) < max_retries:
            job.retry_count = int(job.retry_count or 0) + 1
            delay_s = min(max_delay, base_delay * (2 ** max(0, job.retry_count - 1)))
            job.status = JobStatus.failed_retryable
            job.next_retry_at = datetime.utcnow() + timedelta(seconds=delay_s)
        else:
            job.status = JobStatus.failed_terminal
        job.finished_at = datetime.utcnow()
        job.error_code = code
        job.error_message = message
        job.last_error = message
        session.add(job)
        await session.commit()


async def _dispatch_job(job: GenerationJob, worker_stats: dict, jobs_completed_count_box: list[int]) -> None:
    job_start = time.perf_counter()

    if job.kind == JobKind.index:
        await _mark_job_running(job.id)

        def _on_doc_complete(doc_idx: int, doc_total: int, doc: Document, chunks: int) -> None:
            doc_progress = f"doc({doc_idx}/{doc_total})"
            filename = getattr(doc, "filename", None) or str(doc.id)[:8]
            _banner(_GREEN, "DOC INDEXED", f"{doc_progress}  document_id={doc.id[:8]}...  filename={filename}  chunks={chunks}")

        await run_index_job(job, on_document_complete=_on_doc_complete)

        async with get_session_ctx() as session:
            updated_job = await session.get(GenerationJob, job.id)
            if updated_job and updated_job.result_json:
                try:
                    progress_data = json.loads(updated_job.result_json)
                    progress_data["ok"] = True
                    await _finish_job_completed(job.id, status=JobStatus.completed, result=progress_data)
                except (json.JSONDecodeError, TypeError):
                    await _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True})
            else:
                await _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True})

        elapsed = round(time.perf_counter() - job_start, 1)
        jobs_completed_count_box[0] += 1
        worker_stats["total_time"] += elapsed

        doc_progress = ""
        async with get_session_ctx() as session:
            updated_job = await session.get(GenerationJob, job.id)
            if updated_job and updated_job.result_json:
                try:
                    r = json.loads(updated_job.result_json)
                    chunks_val = r.get("chunks_indexed") or r.get("chunks_embedded") or r.get("chunks_created", 0)
                    if isinstance(chunks_val, (int, float)):
                        worker_stats["total_chunks"] += int(chunks_val)
                    doc_total = r.get("documents_total", 0)
                    doc_done = r.get("documents_processed", 0)
                    if doc_total > 0:
                        doc_progress = f"  doc({doc_done}/{doc_total})  total_chunks={chunks_val}"
                except (json.JSONDecodeError, TypeError):
                    pass

        completed_count, total_active = await _get_job_counts()
        job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count_box[0]})"
        detail_lines = (
            f"{job_progress}  job_id={job.id}  corpus={job.corpus_id}{doc_progress}  time={elapsed}s",
            "status: completed",
        )
        _banner(_GREEN, "INDEX COMPLETE", "\n".join(detail_lines))
        await _check_and_display_all_jobs_done(worker_stats)

    elif job.kind == JobKind.generate:
        await _mark_job_running(job.id)
        result = await run_generate_job(job)
        elapsed = round(time.perf_counter() - job_start, 1)
        if result["final_status"] in (JobStatus.completed, JobStatus.completed_partial):
            await _finish_job_completed(job.id, status=result["final_status"], result=result["payload"])
            meta = result.get("payload", {}).get("meta", {})
            gen_count = meta.get("generated_count", "?")
            target = meta.get("target_count", "?")
            jobs_completed_count_box[0] += 1
            worker_stats["total_time"] += elapsed
            if isinstance(gen_count, (int, float)):
                worker_stats["total_questions"] += int(gen_count)
            completed_count, total_active = await _get_job_counts()
            job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count_box[0]})"
            status_label = result["final_status"].value
            detail_lines = (
                f"{job_progress}  job_id={job.id}  questions={gen_count}/{target}  time={elapsed}s",
                f"status: {status_label}",
            )
            _banner(_GREEN, "GENERATION COMPLETE", "\n".join(detail_lines))
            await _check_and_display_all_jobs_done(worker_stats)
        else:
            await _mark_job_failed(
                job.id,
                code=result.get("error_code") or "JOB_FAILED",
                message=result.get("error_message") or "job_failed",
                retryable=result["final_status"] == JobStatus.failed_retryable,
            )
            detail_lines = (
                f"job_id={job.id}  error={result.get('error_code','?')}  time={elapsed}s",
                f"status: {result['final_status'].value}",
            )
            _banner(_RED, "GENERATION FAILED", "\n".join(detail_lines))

    elif job.kind == JobKind.generate_question_pool:
        await _mark_job_running(job.id)
        job_data = json.loads(job.request_json)
        await generate_question_pool(
            corpus_id=job_data["corpus_id"],
            syllabus=job_data["syllabus"],
            class_level=job_data["class_level"],
            subject=job_data["subject"],
            chapter=job_data.get("chapter", "Unknown Chapter"),
            difficulty=job_data["difficulty"],
            syllabus_version=job_data.get("syllabus_version", "2024"),
            target_count=int(job_data.get("target_count") or 200),
        )
        await _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True, "pool_generated": True})
        elapsed = round(time.perf_counter() - job_start, 1)
        jobs_completed_count_box[0] += 1
        worker_stats["total_time"] += elapsed
        completed_count, total_active = await _get_job_counts()
        job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count_box[0]})"
        detail_lines = (
            f"{job_progress}  job_id={job.id}  chapter={job_data.get('chapter','?')}  time={elapsed}s",
            "status: completed",
        )
        _banner(_GREEN, "POOL GENERATION COMPLETE", "\n".join(detail_lines))
        await _check_and_display_all_jobs_done(worker_stats)

    elif job.kind == JobKind.refresh_question_pool:
        await _mark_job_running(job.id)
        job_data = json.loads(job.request_json)
        await generate_question_pool(
            corpus_id=job_data["corpus_id"],
            syllabus=job_data["syllabus"],
            class_level=job_data["class_level"],
            subject=job_data["subject"],
            chapter=job_data.get("chapter", "Unknown Chapter"),
            difficulty=job_data["difficulty"],
            syllabus_version=job_data.get("syllabus_version", "2024"),
            target_count=int(job_data.get("target_count") or 200),
        )
        await _finish_job_completed(job.id, status=JobStatus.completed, result={"ok": True, "pool_refreshed": True})
        elapsed = round(time.perf_counter() - job_start, 1)
        jobs_completed_count_box[0] += 1
        worker_stats["total_time"] += elapsed
        completed_count, total_active = await _get_job_counts()
        job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count_box[0]})"
        detail_lines = (
            f"{job_progress}  job_id={job.id}  time={elapsed}s",
            "status: completed",
        )
        _banner(_GREEN, "POOL REFRESH COMPLETE", "\n".join(detail_lines))
        await _check_and_display_all_jobs_done(worker_stats)

    elif job.kind == JobKind.admin_generate_questions:
        await _mark_job_running(job.id)
        result = await run_admin_generate_job(job)
        await _finish_job_completed(job.id, status=JobStatus.completed, result=result)
        elapsed = round(time.perf_counter() - job_start, 1)
        jobs_completed_count_box[0] += 1
        worker_stats["total_time"] += elapsed
        if isinstance(result, dict):
            admin_meta = result.get("meta", {})
            admin_gen_count = admin_meta.get("generated_count", 0)
            if isinstance(admin_gen_count, (int, float)):
                worker_stats["total_questions"] += int(admin_gen_count)
        completed_count, total_active = await _get_job_counts()
        job_progress = f"job({completed_count}/{total_active})" if total_active > 0 else f"job({jobs_completed_count_box[0]})"
        detail_lines = (
            f"{job_progress}  job_id={job.id}  time={elapsed}s",
            "status: completed",
        )
        _banner(_GREEN, "ADMIN GENERATION COMPLETE", "\n".join(detail_lines))
        await _check_and_display_all_jobs_done(worker_stats)

    else:
        raise RuntimeError(f"unsupported_job_kind:{job.kind.value}")


async def _process_job(
    job: GenerationJob,
    settings: Settings,
    worker_stats: dict,
    jobs_completed_count_box: list[int],
) -> None:
    timeout = settings.generation_job_timeout_s
    if job.kind == JobKind.index:
        timeout = settings.indexing_job_timeout_s
    await asyncio.wait_for(
        _dispatch_job(job, worker_stats, jobs_completed_count_box),
        timeout=timeout,
    )


async def run_worker(
    job_kinds: list[str] | None,
    concurrency: int,
    worker_id: str | None,
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_db()
    await asyncio.to_thread(ensure_qdrant_payload_indexes)

    if worker_id:
        settings.worker_id = worker_id

    log.info(
        "worker_started",
        env=settings.env,
        embedding_hard_cap_tokens=int(settings.embedding_hard_cap_tokens),
        concurrency=concurrency,
        job_kinds=job_kinds,
    )

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle_signal)
    except NotImplementedError:
        pass

    sem = asyncio.Semaphore(concurrency)
    jobs_completed_count_box = [0]
    worker_stats: dict = {
        "total_questions": 0,
        "total_chunks": 0,
        "total_time": 0.0,
    }

    async def _run_one(job: GenerationJob) -> None:
        async with sem:
            rss_mb = get_process_rss_mb()
            completed_count, total_active = await _get_job_counts()
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
                await _process_job(job, settings, worker_stats, jobs_completed_count_box)
            except asyncio.TimeoutError:
                elapsed_str = f"{settings.generation_job_timeout_s}s"
                if job.kind == JobKind.index:
                    elapsed_str = f"{settings.indexing_job_timeout_s}s"
                log.exception("job_timeout", job_id=job.id, kind=job.kind.value)
                await _mark_job_failed(job.id, code="JOB_TIMEOUT", message=f"Job timed out after {elapsed_str}", retryable=True)
                _banner(_RED, "JOB TIMEOUT", f"job={job.id}  kind={job.kind.value}  timeout={elapsed_str}")
            except RetryableError as e:
                elapsed = round(time.perf_counter(), 1)
                log.exception("job_failed_retryable", job_id=job.id, kind=job.kind.value)
                await _mark_job_failed(job.id, code="JOB_RETRYABLE", message=str(e), retryable=True)
                _banner(_YELLOW, "JOB FAILED (retryable)", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}")
            except TerminalError as e:
                log.exception("job_failed_terminal", job_id=job.id, kind=job.kind.value)
                await _mark_job_failed(job.id, code="JOB_TERMINAL", message=str(e), retryable=False)
                _banner(_RED, "JOB FAILED (terminal)", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}")
            except Exception as e:  # noqa: BLE001
                log.exception("job_failed", job_id=job.id, kind=job.kind.value)
                await _mark_job_failed(job.id, code="JOB_FAILED", message=str(e), retryable=True)
                _banner(_RED, "JOB FAILED", f"job={job.id}  kind={job.kind.value}  error={str(e)[:100]}")

    tasks: set[asyncio.Task] = set()

    _claim_retries = 5
    _claim_retry_delay_s = 10

    while not _shutdown_event.is_set():
        tasks = {t for t in tasks if not t.done()}

        job = None
        for attempt in range(_claim_retries):
            try:
                job = await _claim_next_job(job_kinds)
                break
            except (asyncio.TimeoutError, OSError) as e:
                if attempt + 1 >= _claim_retries:
                    log.exception("claim_next_job_failed", error=str(e), attempt=attempt + 1)
                    raise
                log.warning(
                    "claim_next_job_retry",
                    error=str(e)[:200],
                    attempt=attempt + 1,
                    max_retries=_claim_retries,
                    delay_s=_claim_retry_delay_s,
                )
                await asyncio.sleep(_claim_retry_delay_s)

        if not job:
            await asyncio.sleep(settings.worker_poll_interval_s)
            continue

        task = asyncio.create_task(_run_one(job))
        tasks.add(task)

    if tasks:
        log.info("worker_draining", pending_tasks=len(tasks))
        await asyncio.gather(*tasks, return_exceptions=True)

    log.info("worker_stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Niyukti async job worker")
    parser.add_argument("--job-kinds", type=str, default=None, help="Comma-separated job kinds to process")
    parser.add_argument("--worker-id", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    job_kinds = [k.strip() for k in args.job_kinds.split(",")] if args.job_kinds else None
    asyncio.run(run_worker(job_kinds, args.concurrency, args.worker_id))


if __name__ == "__main__":
    main()

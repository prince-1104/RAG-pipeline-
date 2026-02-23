from __future__ import annotations

import json
import re
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import select

from app.api.v1.schemas.generation import GenerateRequest, GenerateResponse, GeneratedQuestionOut
from app.db.models import Chunk, GeneratedQuestion, JobStatus, QuestionCitation
from app.db.session import get_session
from app.generation.sync import generate_questions_sync
from app.retrieval.retriever import _normalize_subject
from app.observability.logging import log
from app.settings import get_settings


class RetryableError(RuntimeError):
    """A job failure that should be retried."""


class TerminalError(RuntimeError):
    """A job failure that should not be retried."""


def _list_available_chapters(
    *,
    corpus_id: str,
    document_id: str | None,
    class_level: int,
    subject: str,
) -> list[str]:
    normalized_subject = _normalize_subject(subject)
    with get_session() as session:
        rows = session.exec(
            select(Chunk.chapter)
            .where(
                Chunk.corpus_id == corpus_id,
                Chunk.class_level == class_level,
                Chunk.subject == normalized_subject,
                *( [Chunk.document_id == document_id] if document_id else [] ),
            )
            .distinct()
        ).all()
    chapters = [str(r).strip() for r in rows if r and str(r).strip()]
    return chapters or ["Unknown Chapter"]


def _resolve_requested_chapters(req: GenerateRequest, *, document_id: str | None) -> list[str]:
    """
    Normalize requested chapters to match stored chapter strings.

    Example:
    - request chapters: ["9"]
    - stored chapter: "Chapter 9: Light - Reflection"
    -> returns ["Chapter 9: Light - Reflection"]
    """
    requested = [str(c).strip() for c in (req.blueprint.constraints.chapters or []) if str(c).strip()]
    if not requested:
        return requested

    available = _list_available_chapters(
        corpus_id=req.corpus_id,
        document_id=document_id,
        class_level=req.blueprint.constraints.class_level,
        subject=req.blueprint.constraints.subject,
    )

    # If a request chapter matches exactly, keep it.
    exact = set([c for c in requested if c in available])
    if len(exact) == len(requested):
        return requested

    # Patterns to try for numeric chapter resolution (e.g. "1" -> "Chapter 1: ...")
    # Order: canonical first, then common variants (Chapter 1 often appears as "Ch. 1" or "1. Title")
    def _numeric_chapter_patterns(n: int) -> list[str]:
        return [
            f"Chapter {n}%",
            f"Ch. {n}%",
            f"Ch {n}%",
            f"Ch.{n}%",
            f"{n}.%",
            f"{n}:%",
        ]

    resolved: list[str] = []
    for ch in requested:
        # Numeric chapter like "13" or "1" -> resolve to corpus canonical chapter name.
        if re.fullmatch(r"\d+", ch):
            num = int(ch)
            with get_session() as session:
                base = (
                    select(Chunk.chapter)
                    .where(Chunk.corpus_id == req.corpus_id)
                    .where(Chunk.class_level == req.blueprint.constraints.class_level)
                    .where(Chunk.subject == _normalize_subject(req.blueprint.constraints.subject))
                )
                if document_id:
                    base = base.where(Chunk.document_id == document_id)
                matches: list[str] = []
                for pattern in _numeric_chapter_patterns(num):
                    q = base.where(Chunk.chapter.ilike(pattern))
                    rows = session.exec(q.distinct()).all()
                    matches = [str(r).strip() for r in rows if r and str(r).strip()]
                    if matches:
                        break
                # Fallback: match by chapter_number if no text pattern matched
                if len(matches) == 0:
                    q_num = (
                        select(Chunk.chapter)
                        .where(Chunk.corpus_id == req.corpus_id)
                        .where(Chunk.class_level == req.blueprint.constraints.class_level)
                        .where(Chunk.subject == _normalize_subject(req.blueprint.constraints.subject))
                        .where(Chunk.chapter_number == num)
                    )
                    if document_id:
                        q_num = q_num.where(Chunk.document_id == document_id)
                    rows = session.exec(q_num.distinct()).all()
                    matches = [str(r).strip() for r in rows if r and str(r).strip()]
                if len(matches) == 0:
                    available_preview = available[:5] if available else []
                    raise TerminalError(
                        f"chapter_not_found:{ch}. Available for this corpus/subject/class: {available_preview!r}"
                    )
                if len(matches) > 1:
                    raise TerminalError(f"chapter_ambiguous:{ch}")
            resolved.append(matches[0])
            continue

        # "Chapter 9" / "Chapter 9: ..." (exact/starts-with match corpus-wide)
        matches = [a for a in available if a.lower().startswith(ch.lower())]
        if not matches:
            available_preview = available[:5] if available else []
            raise TerminalError(
                f"chapter_not_found:{ch}. Available for this corpus/subject/class: {available_preview!r}"
            )
        if len(matches) > 1 and ch.lower() == "chapter":
            raise TerminalError(f"chapter_ambiguous:{ch}")
        resolved.extend(matches)

    # If every requested chapter was skipped (e.g. none exist in corpus), fail the job
    if not resolved:
        available_preview = available[:5] if available else []
        raise TerminalError(
            f"no_chapters_resolved. Requested: {requested!r}. Available for this corpus/subject/class: {available_preview!r}"
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for ch in resolved:
        if ch in seen:
            continue
        seen.add(ch)
        deduped.append(ch)

    return deduped


def _chunk_counts_by_chapter(req: GenerateRequest, *, document_id: str | None) -> dict[str, int]:
    chapters = list(req.blueprint.constraints.chapters or [])
    if req.blueprint.constraints.per_chapter and len(chapters) <= 1:
        chapters = _list_available_chapters(
            corpus_id=req.corpus_id,
            document_id=document_id,
            class_level=req.blueprint.constraints.class_level,
            subject=req.blueprint.constraints.subject,
        )
    normalized_subject = _normalize_subject(req.blueprint.constraints.subject)
    with get_session() as session:
        rows = session.exec(
            select(Chunk.chapter, func.count(Chunk.id))
            .where(
                Chunk.corpus_id == req.corpus_id,
                Chunk.class_level == req.blueprint.constraints.class_level,
                Chunk.subject == normalized_subject,
                Chunk.chapter.in_(chapters),
                *( [Chunk.document_id == document_id] if document_id else [] ),
            )
            .group_by(Chunk.chapter)
        ).all()
    return {str(ch): int(count) for ch, count in rows}


def _preflight_validate(req: GenerateRequest, *, document_id: str | None) -> bool:
    counts = _chunk_counts_by_chapter(req, document_id=document_id)
    total_chunks = sum(counts.values())
    target = int(req.blueprint.count)
    s = get_settings()
    max_possible = int(total_chunks * int(s.job_questions_per_chunk))
    min_needed = int(target * 0.6)
    log.info(
        "generation_preflight",
        corpus_id=req.corpus_id,
        document_id=document_id,
        class_level=req.blueprint.constraints.class_level,
        subject=req.blueprint.constraints.subject,
        chapters=list(req.blueprint.constraints.chapters),
        chunks=total_chunks,
        max_possible=max_possible,
        target=target,
        min_needed=min_needed,
    )
    if max_possible < min_needed:
        log.warning(
            "generation_preflight_insufficient_content",
            corpus_id=req.corpus_id,
            class_level=req.blueprint.constraints.class_level,
            subject=req.blueprint.constraints.subject,
            target=target,
            chunks=total_chunks,
            max_possible=max_possible,
        )
        return False
    return True


def _build_strategy_requests(req: GenerateRequest) -> list[tuple[str, GenerateRequest]]:
    def _clone() -> GenerateRequest:
        return req.model_copy(deep=True)

    strategies: list[tuple[str, GenerateRequest]] = [("strict_constraints", req)]

    # no_per_chapter: disable per-chapter split, keep requested chapters
    no_per_chapter_req = _clone()
    no_per_chapter_req.blueprint.constraints.per_chapter = False
    strategies.append(("no_per_chapter", no_per_chapter_req))

    # mcq_only: reduce format complexity
    mcq_only_req = _clone()
    mcq_only_req.blueprint.question_types = ["mcq"]
    strategies.append(("mcq_only", mcq_only_req))

    # Deduplicate identical strategy payloads to avoid redundant calls.
    seen: set[str] = set()
    deduped: list[tuple[str, GenerateRequest]] = []
    for name, req_local in strategies:
        key = req_local.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, req_local))
    return deduped


def _scaled_difficulty_mix(req_local: GenerateRequest, target_count: int):
    """Scale difficulty mix to exactly match remaining target_count."""
    mix = req_local.blueprint.difficulty_mix
    base = {"easy": int(mix.easy), "medium": int(mix.medium), "hard": int(mix.hard)}
    total = sum(base.values())
    if target_count <= 0:
        return type(mix)(easy=0, medium=0, hard=0)
    if total <= 0:
        # Fallback to a balanced mix when input is malformed.
        q, r = divmod(target_count, 3)
        vals = {"easy": q, "medium": q, "hard": q}
        for k in ("easy", "medium", "hard")[:r]:
            vals[k] += 1
        return type(mix)(**vals)

    # Proportional split with deterministic remainder assignment.
    scaled = {k: (base[k] * target_count) // total for k in base}
    assigned = sum(scaled.values())
    remainder = target_count - assigned
    order = sorted(base.keys(), key=lambda k: base[k], reverse=True)
    idx = 0
    while remainder > 0:
        scaled[order[idx % len(order)]] += 1
        remainder -= 1
        idx += 1
    return type(mix)(**scaled)


def _dedupe_key_for_question(q: GeneratedQuestionOut) -> str:
    tags = q.tags or {}
    dk = str(tags.get("dedupe_key") or "").strip()
    if dk:
        return dk
    stem = q.stem or ""
    qtype = q.qtype or ""
    chapter = str(tags.get("chapter") or "")
    return f"{qtype}:{chapter}:{stem}".lower().strip()


def _parse_chapter_identity(chapter_value: str | None, fallback_chapter: str | None) -> tuple[int | None, str | None]:
    raw = str(chapter_value or "").strip()
    fallback = str(fallback_chapter or "").strip()
    source = raw or fallback
    if not source:
        return None, None
    if source.isdigit():
        parsed_number = int(source)
        if fallback and not fallback.isdigit():
            fb_num, fb_name = _parse_chapter_identity(fallback, None)
            if fb_num == parsed_number and fb_name:
                return parsed_number, fb_name
        return parsed_number, None

    match = re.match(r"^(?:chapter|unit|ch)\s*(\d+)\s*[:\-]\s*(.+)$", source, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), (match.group(2) or "").strip() or None

    match = re.match(r"^(?:chapter|unit|ch)\s*(\d+)\s*$", source, flags=re.IGNORECASE)
    if match:
        parsed_number = int(match.group(1))
        if fallback and fallback != source:
            fb_num, fb_name = _parse_chapter_identity(fallback, None)
            if fb_num == parsed_number and fb_name:
                return parsed_number, fb_name
        return parsed_number, None

    return None, source


def run_generate_job(job) -> dict:
    """
    Async generation job:
    - applies preflight validation
    - uses adaptive strategies to best-effort hit target
    - returns status + result payload
    """
    raw_req = json.loads(job.request_json)
    req = GenerateRequest.model_validate(raw_req)
    document_id = str(raw_req.get("document_id") or "").strip() or None
    # Normalize chapters (e.g., "9" -> "Chapter 9: ...") so DB filters match.
    resolved_chapters = _resolve_requested_chapters(req, document_id=document_id)
    if resolved_chapters:
        req.blueprint.constraints.chapters = resolved_chapters
    log.info(
        "generation_scope",
        corpus_id=req.corpus_id,
        document_id=document_id,
        scope="document-scoped" if document_id else "corpus-wide",
        chapters=resolved_chapters,
    )
    _preflight_validate(req, document_id=document_id)

    target = int(req.blueprint.count)
    min_needed = max(1, int(target * 0.6))
    strategies = _build_strategy_requests(req)
    generated: list[GeneratedQuestionOut] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    attempted: list[str] = []
    last_error: Exception | None = None

    for name, req_local in strategies:
        deficit = max(0, target - len(generated))
        if deficit <= 0:
            break
        req_local = req_local.model_copy(deep=True)
        req_local.blueprint.count = deficit
        req_local.blueprint.difficulty_mix = _scaled_difficulty_mix(req_local, deficit)
        attempted.append(name)
        try:
            resp: GenerateResponse = generate_questions_sync(req_local, document_id=document_id)
        except HTTPException as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        if resp.rejected:
            rejected.extend(resp.rejected)
        if resp.questions:
            for q in resp.questions:
                dk = _dedupe_key_for_question(q)
                if dk in seen:
                    continue
                seen.add(dk)
                generated.append(q)

        if len(generated) >= target:
            break
        # Accept partial: one LLM run per chapter; do not retry with other strategies for deficit.
        if len(generated) >= min_needed:
            break

    generated = generated[:target]
    ratio = (len(generated) / target) if target > 0 else 0.0

    if ratio >= 1.0:
        final_status = JobStatus.completed
    elif len(generated) > 0 and ratio >= 0.6:
        final_status = JobStatus.completed_partial
    else:
        # Under-threshold completion is terminal to avoid retry thrash loops.
        final_status = JobStatus.failed_terminal

    # Persist generated questions + citations for auditability (institutes care).
    if document_id:
        with get_session() as session:
            db_job = session.get(type(job), job.id)
            if db_job:
                db_job.document_id = document_id
                session.add(db_job)
                session.commit()

    if generated:
        with get_session() as session:
            # Build all question rows first, flush once to assign IDs.
            all_questions: list[GeneratedQuestion] = []
            for q in generated:
                tags = q.tags or {}
                chapter_value = str(tags.get("chapter") or req.blueprint.constraints.chapters[0])
                chapter_number, chapter_name = _parse_chapter_identity(
                    chapter_value,
                    req.blueprint.constraints.chapters[0],
                )
                all_questions.append(
                    GeneratedQuestion(
                        job_id=job.id,
                        corpus_id=req.corpus_id,
                        document_id=document_id,
                        class_level=req.blueprint.constraints.class_level,
                        subject=req.blueprint.constraints.subject,
                        chapter=chapter_value,
                        chapter_number=chapter_number,
                        chapter_name=chapter_name,
                        difficulty=q.difficulty,
                        qtype=q.qtype,
                        language=q.language,
                        question_json=q.model_dump_json(),
                        dedupe_key=str(tags.get("dedupe_key") or ""),
                    )
                )

            session.add_all(all_questions)
            session.flush()  # assigns IDs without committing

            # Only persist citations for chunks that exist in DB (Qdrant can return stale/missing chunk IDs).
            citation_chunk_ids = {cit.chunk_id for q in generated for cit in q.citations}
            valid_chunk_query = (
                select(Chunk.id).where(Chunk.corpus_id == req.corpus_id)
            )
            if document_id:
                valid_chunk_query = valid_chunk_query.where(Chunk.document_id == document_id)
            if citation_chunk_ids:
                valid_chunk_query = valid_chunk_query.where(Chunk.id.in_(citation_chunk_ids))
            valid_chunk_ids = set(
                session.exec(valid_chunk_query).all() if citation_chunk_ids else set()
            )

            all_citations: list[QuestionCitation] = []
            skipped = 0
            for dq, q in zip(all_questions, generated):
                for cit in q.citations:
                    if cit.chunk_id not in valid_chunk_ids:
                        skipped += 1
                        continue
                    all_citations.append(
                        QuestionCitation(
                            question_id=dq.id,
                            chunk_id=cit.chunk_id,
                            document_id=cit.document_id,
                            storage_uri=cit.storage_uri,
                            page_start=cit.page_start,
                            page_end=cit.page_end,
                            excerpt=cit.excerpt,
                        )
                    )
            if skipped:
                log.warning(
                    "citations_skipped_missing_chunk",
                    job_id=job.id,
                    skipped=skipped,
                    total_citations=sum(len(q.citations) for q in generated),
                )

            if all_citations:
                session.add_all(all_citations)

            session.commit()

        log.info(
            "generation_persisted",
            job_id=job.id,
            questions=len(generated),
            status=final_status.value,
        )

    if ratio < 0.6 and last_error is not None:
        log.warning(
            "generation_under_threshold",
            job_id=job.id,
            error=str(last_error)[:200],
            ratio=ratio,
            generated=len(generated),
            target=target,
        )

    payload = {
        **GenerateResponse(job_id=job.id, questions=generated, rejected=rejected).model_dump(mode="json"),
        "meta": {
            "target_count": target,
            "generated_count": len(generated),
            "ratio": ratio,
            "strategies_attempted": attempted,
            "completed_at": datetime.utcnow().isoformat(),
        },
    }
    error_message = None
    error_code = None
    if final_status in (JobStatus.failed_retryable, JobStatus.failed_terminal):
        error_code = "GENERATION_UNDER_THRESHOLD"
        error_message = f"generated_{len(generated)}_of_{target}"
        if last_error is not None:
            error_message = f"{error_message}: {str(last_error)[:200]}"

    return {
        "final_status": final_status,
        "payload": payload,
        "error_code": error_code,
        "error_message": error_message,
    }



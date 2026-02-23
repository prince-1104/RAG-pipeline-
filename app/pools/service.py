from __future__ import annotations

import json
import random
import re
from datetime import datetime
from typing import Literal

from app.cache.redis_client import get_redis_client
from app.db.models import (
    Chunk,
    GeneratedQuestion,
    PrewarmRun,
    PrewarmRunQuestion,
    PoolQuestion,
    QuestionCitation,
    QuestionPool,
    QuestionPoolStatus,
)
from app.db.session import get_session
from app.observability.logging import log
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import select


def get_pool_key(
    syllabus: str,
    class_level: int,
    subject: str,
    difficulty: str,
    syllabus_version: str,
) -> str:
    """Generate Redis key for a question pool."""
    return f"qp:{syllabus}:{class_level}:{subject}:{difficulty}:{syllabus_version}"


def _parse_chapter_number(chapter: str | None) -> int | None:
    if not chapter:
        return None
    chapter_str = str(chapter).strip()
    if not chapter_str:
        return None
    if chapter_str.isdigit():
        return int(chapter_str)
    match = re.search(r"(?:chapter|unit|ch)\s*(\d+)", chapter_str.lower())
    if match:
        return int(match.group(1))
    return None


def _build_chapter_display(chapter_number: int, chapter_name: str | None) -> str:
    name = (chapter_name or "").strip()
    if name:
        if name.lower().startswith(f"chapter {chapter_number}".lower()):
            return name
        return f"Chapter {chapter_number}: {name}"
    return f"Chapter {chapter_number}"


def _resolve_chapter_identity(
    *,
    chapter: str | None,
    chapter_number: int | None,
    chapter_name: str | None,
) -> tuple[int, str, str]:
    num = int(chapter_number) if chapter_number is not None else _parse_chapter_number(chapter)
    name = (chapter_name or "").strip() or None
    if not name:
        src = (chapter or "").strip()
        # If the source already looks like "Chapter N: Name", extract display name.
        if src:
            m = re.match(r"^(?:chapter|unit|ch)\s*\d+\s*[:\-]\s*(.+)$", src.strip(), flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip() or None
    if num is None:
        raise RuntimeError("missing_chapter_number_for_pool_identity")
    display = _build_chapter_display(num, name)
    return num, name or f"Chapter {num}", display


def start_prewarm_run(*, syllabus: str, syllabus_version: str) -> PrewarmRun:
    """Create and persist a prewarm run record."""
    with get_session() as session:
        run = PrewarmRun(
            status="running",
            syllabus=syllabus,
            syllabus_version=syllabus_version,
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def finish_prewarm_run(
    *,
    prewarm_id: str,
    promoted_count: int,
    pools_created: int,
) -> None:
    with get_session() as session:
        run = session.get(PrewarmRun, prewarm_id)
        if not run:
            return
        run.status = "completed"
        run.promoted_count = int(promoted_count)
        run.pools_created = int(pools_created)
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()


def fail_prewarm_run(*, prewarm_id: str, error_message: str) -> None:
    with get_session() as session:
        run = session.get(PrewarmRun, prewarm_id)
        if not run:
            return
        run.status = "failed"
        run.error_message = str(error_message)[:1000]
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()


def record_prewarm_promoted_questions(*, prewarm_id: str, question_ids: list[str]) -> None:
    if not question_ids:
        return

    def _normalize_id(qid: str) -> str:
        if hasattr(qid, "_mapping"):
            return str(next(iter(qid._mapping.values())))
        if isinstance(qid, (tuple, list)):
            return str(qid[0])
        return str(qid)

    for qid in question_ids:
        q = _normalize_id(qid)
        assert "(" not in q and ")" not in q, f"Invalid question_id: {q}"

    rows = [
        {
            "prewarm_id": prewarm_id,
            "question_id": _normalize_id(qid),
        }
        for qid in question_ids
    ]
    with get_session() as session:
        insert_stmt = (
            pg_insert(PrewarmRunQuestion)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["prewarm_id", "question_id"])
        )
        session.exec(insert_stmt)
        session.commit()


def get_prewarm_promoted_questions(*, prewarm_id: str) -> list[dict]:
    """Fetch canonical payload for questions promoted in a prewarm run."""
    with get_session() as session:
        run = session.get(PrewarmRun, prewarm_id)
        if not run:
            return []

        mapped_ids = session.exec(
            select(PrewarmRunQuestion.question_id).where(PrewarmRunQuestion.prewarm_id == prewarm_id)
        ).all()
        ordered_ids = [str(qid) for qid in mapped_ids]
        if not ordered_ids:
            return []

        pool_rows = session.exec(
            select(PoolQuestion, QuestionPool)
            .join(QuestionPool, QuestionPool.id == PoolQuestion.pool_id)
            .where(PoolQuestion.id.in_(ordered_ids))
        ).all()
        generated_rows = session.exec(
            select(GeneratedQuestion).where(GeneratedQuestion.id.in_(ordered_ids))
        ).all()
        citation_rows = session.exec(
            select(QuestionCitation).where(QuestionCitation.question_id.in_(ordered_ids))
        ).all()

    pool_by_qid: dict[str, tuple[PoolQuestion, QuestionPool]] = {}
    for pq, pool in pool_rows:
        pool_by_qid[str(pq.id)] = (pq, pool)
    generated_by_qid = {str(gq.id): gq for gq in generated_rows}
    citation_doc_by_qid: dict[str, str] = {}
    for cit in citation_rows:
        qid = str(cit.question_id)
        if qid not in citation_doc_by_qid and cit.document_id:
            citation_doc_by_qid[qid] = str(cit.document_id)

    out: list[dict] = []
    for qid in ordered_ids:
        pair = pool_by_qid.get(qid)
        if not pair:
            continue
        pq, pool = pair
        gq = generated_by_qid.get(qid)
        payload = json.loads(pq.question_json) if pq.question_json else {}
        chapter_number = pool.chapter_number
        if chapter_number is None:
            raise RuntimeError(f"missing_chapter_number_for_promoted_question:{qid}")
        source_document_id = ""
        rag_generation_job_id = ""
        question_type = ""
        if gq:
            source_document_id = str(gq.document_id or "")
            rag_generation_job_id = str(gq.job_id or "")
            question_type = str(gq.qtype or "")
        if not source_document_id:
            source_document_id = citation_doc_by_qid.get(qid, "")
        if not question_type:
            question_type = str(payload.get("qtype") or "")

        out.append(
            {
                "question_id": qid,
                "board": str(pool.board or pool.syllabus or "CBSE"),
                "class_level": int(pool.class_level),
                "subject": str(pool.subject),
                "syllabus_version": str(pool.syllabus_version),
                "chapter_number": int(chapter_number),
                "difficulty": str(pq.difficulty or pool.difficulty),
                "question_type": question_type,
                "question_json": payload if isinstance(payload, dict) else {},
                "source_document_id": source_document_id,
                "rag_generation_job_id": rag_generation_job_id,
            }
        )

    return out


def list_pool_chapters(corpus_id: str, class_level: int, subject: str) -> list[str]:
    """List distinct chapters for a corpus/class/subject."""
    with get_session() as session:
        rows = session.exec(
            select(Chunk.chapter).where(
                Chunk.corpus_id == corpus_id,
                Chunk.class_level == class_level,
                Chunk.subject == subject,
            )
        ).all()
    chapters = [str(r).strip() for r in rows if r and str(r).strip()]
    if not chapters:
        return ["Unknown Chapter"]
    # Preserve insertion order, de-dup
    seen: set[str] = set()
    out: list[str] = []
    for ch in chapters:
        if ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def get_or_create_pool(
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
    difficulty: str,
    syllabus_version: str = "2024",
    *,
    board: str | None = None,
    chapter_number: int | None = None,
    chapter_name: str | None = None,
) -> QuestionPool:
    """Get existing pool or create a new one in pending status."""
    resolved_number, resolved_name, resolved_chapter = _resolve_chapter_identity(
        chapter=chapter,
        chapter_number=chapter_number,
        chapter_name=chapter_name,
    )
    resolved_board = (board or syllabus or "CBSE").strip()
    with get_session() as session:
        pool = session.exec(
            select(QuestionPool).where(
                QuestionPool.syllabus == syllabus,
                QuestionPool.board == resolved_board,
                QuestionPool.class_level == class_level,
                QuestionPool.subject == subject,
                QuestionPool.chapter_number == resolved_number,
                QuestionPool.difficulty == difficulty,
                QuestionPool.syllabus_version == syllabus_version,
            )
        ).first()
        
        if not pool:
            pool = QuestionPool(
                syllabus=syllabus,
                board=resolved_board,
                class_level=class_level,
                subject=subject,
                chapter=resolved_chapter,
                chapter_number=resolved_number,
                chapter_name=resolved_name,
                difficulty=difficulty,
                syllabus_version=syllabus_version,
                status=QuestionPoolStatus.pending,
            )
            session.add(pool)
            session.commit()
            session.refresh(pool)
            log.info(
                "question_pool_created",
                pool_id=pool.id,
                key=get_pool_key(syllabus, class_level, subject, difficulty, syllabus_version),
            )
        
        return pool


def get_pool_status(
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
    difficulty: str,
    syllabus_version: str = "2024",
    *,
    board: str | None = None,
    chapter_number: int | None = None,
) -> QuestionPoolStatus | None:
    """Get pool status from database."""
    resolved_board = (board or syllabus or "CBSE").strip()
    resolved_number = int(chapter_number) if chapter_number is not None else _parse_chapter_number(chapter)
    with get_session() as session:
        stmt = select(QuestionPool).where(
            QuestionPool.syllabus == syllabus,
            QuestionPool.board == resolved_board,
            QuestionPool.class_level == class_level,
            QuestionPool.subject == subject,
            QuestionPool.difficulty == difficulty,
            QuestionPool.syllabus_version == syllabus_version,
        )
        if resolved_number is not None:
            stmt = stmt.where(QuestionPool.chapter_number == resolved_number)
        else:
            stmt = stmt.where(QuestionPool.chapter == chapter)
        pool = session.exec(stmt).first()
        return pool.status if pool else None


def get_questions_from_pool(
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
    difficulty: str,
    syllabus_version: str,
    count: int,
) -> list[PoolQuestion]:
    """
    Fetch questions from Redis pool (fast path).
    Returns empty list if pool is not ready or insufficient questions.
    """
    redis_client = get_redis_client()
    pool_key = get_pool_key(syllabus, class_level, subject, difficulty, syllabus_version)
    
    # Check if pool exists and is ready
    pool_status = get_pool_status(syllabus, class_level, subject, chapter, difficulty, syllabus_version)
    if pool_status not in (QuestionPoolStatus.ready, QuestionPoolStatus.ready_partial):
        log.warning("pool_not_ready", pool_key=pool_key, status=pool_status)
        return []
    
    # Get question IDs from Redis (pre-shuffled list)
    question_ids = redis_client.lrange(pool_key, 0, count - 1)
    available = len(question_ids)
    if available < count and pool_status == QuestionPoolStatus.ready:
        # strict for fully-ready pools (should have enough); trigger refresh upstream
        log.warning("pool_insufficient_questions", pool_key=pool_key, requested=count, available=available)
        return []
    if available == 0:
        log.warning("pool_empty", pool_key=pool_key)
        return []
    
    # Fetch questions from database
    with get_session() as session:
        questions = session.exec(
            select(PoolQuestion).where(PoolQuestion.id.in_(question_ids))
        ).all()
        
        # Remove fetched IDs from Redis (pop from left)
        redis_client.ltrim(pool_key, min(count, available), -1)
        
        log.info("questions_fetched_from_pool", pool_key=pool_key, count=len(questions))
        return list(questions)


def get_questions_from_difficulty_pool(
    syllabus: str,
    class_level: int,
    subject: str,
    difficulty: str,
    syllabus_version: str,
    count: int,
) -> list[PoolQuestion]:
    """
    Fetch questions from a difficulty-level Redis pool (shared across chapters).
    Does not check per-chapter pool status; caller must enforce readiness.
    """
    redis_client = get_redis_client()
    pool_key = get_pool_key(syllabus, class_level, subject, difficulty, syllabus_version)
    question_ids = redis_client.lrange(pool_key, 0, count - 1)
    available = len(question_ids)
    if available == 0:
        log.warning("pool_empty", pool_key=pool_key)
        return []

    with get_session() as session:
        questions = session.exec(
            select(PoolQuestion).where(PoolQuestion.id.in_(question_ids))
        ).all()

        redis_client.ltrim(pool_key, min(count, available), -1)
        log.info("questions_fetched_from_pool", pool_key=pool_key, count=len(questions))
        return list(questions)


def add_questions_to_pool(
    pool_id: str,
    question_ids: list[str],
    *,
    requested_count: int | None = None,
    generated_count: int | None = None,
    status: QuestionPoolStatus | None = None,
    last_error: str | None = None,
) -> None:
    """
    Add question IDs to Redis pool list (pre-shuffled).
    Called after pool generation completes.
    """
    with get_session() as session:
        pool = session.get(QuestionPool, pool_id)
        if not pool:
            log.error("pool_not_found", pool_id=pool_id)
            return
        
        pool_key = get_pool_key(
            pool.syllabus,
            pool.class_level,
            pool.subject,
            pool.difficulty,
            pool.syllabus_version,
        )
        
        redis_client = get_redis_client()
        
        # Shuffle and add to Redis list
        shuffled_ids = question_ids.copy()
        random.shuffle(shuffled_ids)

        # Enforce requested_count cap if provided (do not exceed)
        effective_requested = requested_count if requested_count is not None else pool.requested_count
        current_generated = pool.generated_count or 0
        if generated_count is not None:
            current_generated = max(current_generated, generated_count)
        if effective_requested is not None:
            remaining = int(effective_requested) - int(current_generated)
            if remaining <= 0:
                log.info(
                    "pool_generation_cap_reached",
                    pool_id=pool.id,
                    requested=effective_requested,
                    generated=current_generated,
                )
                shuffled_ids = []
            elif len(shuffled_ids) > remaining:
                shuffled_ids = shuffled_ids[:remaining]
        
        if shuffled_ids:
            redis_client.rpush(pool_key, *shuffled_ids)
            # Set long TTL (30 days) - invalidate manually on syllabus change
            redis_client.expire(pool_key, 30 * 24 * 60 * 60)
            log.info("questions_added_to_pool", pool_key=pool_key, count=len(shuffled_ids))
        
        # Update pool status (default: ready)
        pool.status = status or QuestionPoolStatus.ready
        if requested_count is not None:
            pool.requested_count = requested_count
        if generated_count is not None:
            pool.generated_count = generated_count
        elif shuffled_ids:
            pool.generated_count = (pool.generated_count or 0) + len(shuffled_ids)
        if last_error is not None:
            pool.last_error = last_error
        session.add(pool)
        session.commit()


def mark_pool_generating(pool_id: str) -> None:
    """Mark pool as generating."""
    with get_session() as session:
        pool = session.get(QuestionPool, pool_id)
        if pool:
            pool.status = QuestionPoolStatus.generating
            session.add(pool)
            session.commit()


def mark_pool_failed(pool_id: str, error_message: str | None = None) -> None:
    """Mark pool generation as failed."""
    with get_session() as session:
        pool = session.get(QuestionPool, pool_id)
        if pool:
            pool.status = QuestionPoolStatus.failed
            pool.last_error = error_message
            session.add(pool)
            session.commit()
            log.error("pool_generation_failed", pool_id=pool_id, error=error_message)


def mark_pool_ready_partial(pool_id: str, *, requested_count: int, generated_count: int, last_error: str | None = None) -> None:
    """Mark pool as ready_partial and persist counts."""
    with get_session() as session:
        pool = session.get(QuestionPool, pool_id)
        if pool:
            pool.status = QuestionPoolStatus.ready_partial
            pool.requested_count = requested_count
            pool.generated_count = generated_count
            pool.last_error = last_error
            session.add(pool)
            session.commit()


def promote_generated_questions_to_pools(
    *,
    document_id: str,
    class_level: int,
    subject: str,
    syllabus: str,
    syllabus_version: str,
    difficulty: list[str] | None = None,
    question_types: list[str] | None = None,
) -> dict:
    """
    PROMOTION ONLY – NO GENERATION.
    Promote GeneratedQuestion rows to PoolQuestion + Redis pools using DB + Redis ONLY.
    Returns summary {status, pools_created, questions_promoted, reason?}
    """
    def _normalize_ids(items: list) -> set[str]:
        out: set[str] = set()
        for item in items:
            if isinstance(item, bytes):
                out.add(item.decode("utf-8", errors="ignore"))
            else:
                out.add(str(item))
        return out

    def _prewarm_pool_redis(*, pool: QuestionPool, new_question_ids: list[str]) -> int:
        """
        PROMOTION ONLY – NO GENERATION.
        Prewarm Redis with pool question IDs without overwriting existing data.
        """
        redis_client = get_redis_client()
        pool_key = get_pool_key(
            pool.syllabus,
            pool.class_level,
            pool.subject,
            pool.difficulty,
            pool.syllabus_version,
        )

        existing_count = int(redis_client.llen(pool_key))
        existing_ids = _normalize_ids(redis_client.lrange(pool_key, 0, -1)) if existing_count else set()

        ids_to_add: list[str] = []
        if new_question_ids:
            ids_to_add = [qid for qid in new_question_ids if qid not in existing_ids]
        else:
            # Redis missing/partial: add any DB IDs not yet present.
            with get_session() as session:
                db_ids = session.exec(
                    select(PoolQuestion.id).where(PoolQuestion.pool_id == pool.id)
                ).all()
            db_set = _normalize_ids(db_ids)
            ids_to_add = [qid for qid in db_set if qid not in existing_ids]

        if not ids_to_add:
            return 0

        random.shuffle(ids_to_add)
        redis_client.rpush(pool_key, *ids_to_add)
        redis_client.expire(pool_key, 30 * 24 * 60 * 60)
        log.info("redis_pool_prewarmed", pool_key=pool_key, count=len(ids_to_add))
        return len(ids_to_add)

    with get_session() as session:
        stmt = (
            select(GeneratedQuestion)
            .join(QuestionCitation, QuestionCitation.question_id == GeneratedQuestion.id)
            .where(QuestionCitation.document_id == document_id)
            .where(GeneratedQuestion.class_level == class_level)
            .where(GeneratedQuestion.subject == subject)
            .distinct()
        )
        if difficulty:
            stmt = stmt.where(GeneratedQuestion.difficulty.in_(difficulty))
        if question_types:
            stmt = stmt.where(GeneratedQuestion.qtype.in_(question_types))

        generated = session.exec(stmt).all()

    if not generated:
        return {"status": "noop", "reason": "no_generated_questions", "pools_created": 0, "questions_promoted": 0}

    groups: dict[tuple[str, str, int, str, int, str, str], list[GeneratedQuestion]] = {}
    for gq in generated:
        chapter_number = _parse_chapter_number(gq.chapter)
        if chapter_number is None:
            raise RuntimeError(f"missing_chapter_number_for_generated_question:{gq.id}")
        board = (syllabus or "CBSE").strip()
        key = (
            board,
            syllabus_version,
            gq.class_level,
            gq.subject,
            int(chapter_number),
            gq.difficulty,
            gq.qtype,
        )
        groups.setdefault(key, []).append(gq)

    pools_created = 0
    questions_promoted = 0

    with get_session() as session:
        for (board, _syllabus_version, cl, subj, chapter_num, diff, _qtype), items in groups.items():
            # PROMOTION ONLY – NO GENERATION.
            pool = session.exec(
                select(QuestionPool).where(
                    QuestionPool.syllabus == syllabus,
                    QuestionPool.board == board,
                    QuestionPool.class_level == cl,
                    QuestionPool.subject == subj,
                    QuestionPool.chapter_number == chapter_num,
                    QuestionPool.difficulty == diff,
                    QuestionPool.syllabus_version == _syllabus_version,
                )
            ).first()

            resolved_chapter_name: str | None = None

            if not pool:
                chapter_name = f"Chapter {chapter_num}"
                pool = QuestionPool(
                    syllabus=syllabus,
                    board=board,
                    class_level=cl,
                    subject=subj,
                    chapter=_build_chapter_display(chapter_num, chapter_name),
                    chapter_number=chapter_num,
                    chapter_name=chapter_name,
                    difficulty=diff,
                    syllabus_version=_syllabus_version,
                    status=QuestionPoolStatus.ready,
                )
                session.add(pool)
                session.commit()
                session.refresh(pool)
                pools_created += 1

            rows = []
            # Validate source_chunk_id to avoid FK violations during promotion.
            candidate_chunk_ids: set[str] = set()
            for gq in items:
                try:
                    q_data = json.loads(gq.question_json)
                except (TypeError, json.JSONDecodeError):
                    continue
                citations = q_data.get("citations") or []
                if citations:
                    cid = citations[0].get("chunk_id")
                    if cid:
                        candidate_chunk_ids.add(str(cid))

            existing_chunk_ids: set[str] = set()
            existing_chunks_by_id: dict[str, Chunk] = {}
            if candidate_chunk_ids:
                existing_chunks = session.exec(
                    select(Chunk).where(Chunk.id.in_(list(candidate_chunk_ids)))
                ).all()
                existing_chunks_by_id = {str(ch.id): ch for ch in existing_chunks}
                existing_chunk_ids = set(existing_chunks_by_id.keys())

            for gq in items:
                try:
                    q_data = json.loads(gq.question_json)
                except (TypeError, json.JSONDecodeError):
                    log.warning("promotion_skip_invalid_question_json", question_id=gq.id)
                    continue

                citations = q_data.get("citations") or []
                source_chunk_id = citations[0].get("chunk_id") if citations else None
                if source_chunk_id and str(source_chunk_id) not in existing_chunk_ids:
                    log.warning(
                        "promotion_missing_chunk",
                        question_id=gq.id,
                        missing_chunk_id=str(source_chunk_id),
                    )
                    source_chunk_id = None
                if source_chunk_id:
                    source_chunk = existing_chunks_by_id.get(str(source_chunk_id))
                    if source_chunk and source_chunk.chapter_number is not None:
                        if int(source_chunk.chapter_number) != int(chapter_num):
                            raise RuntimeError(
                                f"inconsistent_chapter_number_for_promotion:question={gq.id}"
                            )
                    if source_chunk and not resolved_chapter_name:
                        resolved_chapter_name = (
                            (source_chunk.chapter_name or "").strip()
                            or (source_chunk.chapter or "").strip()
                            or None
                        )
                tags = q_data.get("tags") or {}

                rows.append(
                    {
                        "id": str(gq.id),
                        "pool_id": pool.id,
                        "question_text": q_data.get("stem", ""),
                        "options": json.dumps(q_data.get("options") or []),
                        "correct_answer": q_data.get("answer"),
                        "concept": tags.get("concept"),
                        "difficulty": gq.difficulty,
                        "source_chunk_id": source_chunk_id,
                        "question_json": json.dumps(q_data),
                    }
                )

            if not rows:
                continue

            insert_stmt = (
                pg_insert(PoolQuestion)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["id"])
                .returning(PoolQuestion.id)
            )
            inserted = session.exec(insert_stmt).all()
            session.commit()

            inserted_ids = [str(row[0] if isinstance(row, tuple) else row) for row in inserted]
            if inserted_ids:
                questions_promoted += len(inserted_ids)

            # Ensure pool is marked ready after promotion.
            if resolved_chapter_name and (
                (pool.chapter_name or "").strip() != resolved_chapter_name
                or int(pool.chapter_number or 0) != int(chapter_num)
                or (pool.chapter or "").strip() != _build_chapter_display(chapter_num, resolved_chapter_name)
            ):
                pool.chapter_name = resolved_chapter_name
                pool.chapter_number = chapter_num
                pool.chapter = _build_chapter_display(chapter_num, resolved_chapter_name)
            if pool.status != QuestionPoolStatus.ready:
                pool.status = QuestionPoolStatus.ready
            session.add(pool)
            session.commit()

            _prewarm_pool_redis(pool=pool, new_question_ids=inserted_ids)

    return {
        "status": "success",
        "pools_created": pools_created,
        "questions_promoted": questions_promoted,
    }


def promote_all_generated_questions_to_pools(
    *,
    syllabus: str,
    syllabus_version: str,
    prewarm_id: str | None = None,
) -> dict:
    """
    PROMOTION ONLY – NO GENERATION.
    Promote ALL GeneratedQuestion rows to PoolQuestion + Redis pools using DB + Redis ONLY.
    Returns summary {status, total_generated_questions, questions_promoted, pools_created}
    """
    def _normalize_ids(items: list) -> set[str]:
        out: set[str] = set()
        for item in items:
            if isinstance(item, bytes):
                out.add(item.decode("utf-8", errors="ignore"))
            else:
                out.add(str(item))
        return out

    def _prewarm_pool_redis(*, pool: QuestionPool, new_question_ids: list[str]) -> int:
        """
        PROMOTION ONLY – NO GENERATION.
        Prewarm Redis with pool question IDs without overwriting existing data.
        """
        redis_client = get_redis_client()
        pool_key = get_pool_key(
            pool.syllabus,
            pool.class_level,
            pool.subject,
            pool.difficulty,
            pool.syllabus_version,
        )

        existing_count = int(redis_client.llen(pool_key))
        existing_ids = _normalize_ids(redis_client.lrange(pool_key, 0, -1)) if existing_count else set()

        ids_to_add: list[str] = []
        if new_question_ids:
            ids_to_add = [qid for qid in new_question_ids if qid not in existing_ids]
        else:
            # Redis missing/partial: add any DB IDs not yet present.
            with get_session() as session:
                db_ids = session.exec(
                    select(PoolQuestion.id).where(PoolQuestion.pool_id == pool.id)
                ).all()
            db_set = _normalize_ids(db_ids)
            ids_to_add = [qid for qid in db_set if qid not in existing_ids]

        if not ids_to_add:
            return 0

        random.shuffle(ids_to_add)
        redis_client.rpush(pool_key, *ids_to_add)
        redis_client.expire(pool_key, 30 * 24 * 60 * 60)
        log.info("redis_pool_prewarmed", pool_key=pool_key, count=len(ids_to_add))
        return len(ids_to_add)

    with get_session() as session:
        generated = session.exec(
            select(GeneratedQuestion)
            .outerjoin(PoolQuestion, PoolQuestion.id == GeneratedQuestion.id)
            .where(PoolQuestion.id.is_(None))
        ).all()

    if not generated:
        return {
            "status": "success",
            "questions_promoted": 0,
            "pools_created": 0,
            "promoted_question_ids": [],
        }

    groups: dict[tuple[str, str, int, str, int, str, str], list[GeneratedQuestion]] = {}
    for gq in generated:
        chapter_number = _parse_chapter_number(gq.chapter)
        if chapter_number is None:
            raise RuntimeError(f"missing_chapter_number_for_generated_question:{gq.id}")
        key = (
            syllabus,
            syllabus_version,
            gq.class_level,
            gq.subject,
            int(chapter_number),
            gq.difficulty,
            gq.qtype,
        )
        groups.setdefault(key, []).append(gq)

    pools_created = 0
    questions_promoted = 0
    promoted_question_ids: list[str] = []

    with get_session() as session:
        for (_syllabus, _syllabus_version, cl, subj, chapter_num, diff, _qtype), items in groups.items():
            # PROMOTION ONLY – NO GENERATION.
            resolved_chapter_name: str | None = None
            pool_chapter_name = f"Chapter {chapter_num}"
            pool_chapter = _build_chapter_display(chapter_num, pool_chapter_name)
            pool = session.exec(
                select(QuestionPool).where(
                    QuestionPool.syllabus == _syllabus,
                    QuestionPool.board == _syllabus,
                    QuestionPool.class_level == cl,
                    QuestionPool.subject == subj,
                    QuestionPool.chapter_number == chapter_num,
                    QuestionPool.difficulty == diff,
                    QuestionPool.syllabus_version == _syllabus_version,
                )
            ).first()

            if not pool:
                pool = QuestionPool(
                    syllabus=_syllabus,
                    board=_syllabus,
                    class_level=cl,
                    subject=subj,
                    chapter=pool_chapter,
                    chapter_number=chapter_num,
                    chapter_name=pool_chapter_name,
                    difficulty=diff,
                    syllabus_version=_syllabus_version,
                    status=QuestionPoolStatus.ready,
                )
                session.add(pool)
                session.commit()
                session.refresh(pool)
                pools_created += 1

            rows = []
            # Pre-validate cited chunk ids so we never insert invalid FKs.
            candidate_chunk_ids: set[str] = set()
            for gq in items:
                try:
                    q_data = json.loads(gq.question_json)
                except (TypeError, json.JSONDecodeError):
                    continue
                citations = q_data.get("citations") or []
                if citations:
                    cid = citations[0].get("chunk_id")
                    if cid:
                        candidate_chunk_ids.add(str(cid))

            existing_chunk_ids: set[str] = set()
            existing_chunks_by_id: dict[str, Chunk] = {}
            if candidate_chunk_ids:
                existing_chunks = session.exec(
                    select(Chunk).where(Chunk.id.in_(list(candidate_chunk_ids)))
                ).all()
                existing_chunks_by_id = {str(ch.id): ch for ch in existing_chunks}
                existing_chunk_ids = set(existing_chunks_by_id.keys())

            for gq in items:
                try:
                    q_data = json.loads(gq.question_json)
                except (TypeError, json.JSONDecodeError):
                    log.warning("promotion_skip_invalid_question_json", question_id=gq.id)
                    continue

                citations = q_data.get("citations") or []
                source_chunk_id = citations[0].get("chunk_id") if citations else None
                if source_chunk_id and str(source_chunk_id) not in existing_chunk_ids:
                    log.warning(
                        "promotion_missing_chunk",
                        question_id=gq.id,
                        missing_chunk_id=str(source_chunk_id),
                    )
                    source_chunk_id = None
                if source_chunk_id:
                    source_chunk = existing_chunks_by_id.get(str(source_chunk_id))
                    if source_chunk and source_chunk.chapter_number is not None:
                        if int(source_chunk.chapter_number) != int(chapter_num):
                            raise RuntimeError(
                                f"inconsistent_chapter_number_for_promotion:question={gq.id}"
                            )
                    if source_chunk and not resolved_chapter_name:
                        resolved_chapter_name = (
                            (source_chunk.chapter_name or "").strip()
                            or (source_chunk.chapter or "").strip()
                            or None
                        )
                tags = q_data.get("tags") or {}

                rows.append(
                    {
                        "id": str(gq.id),
                        "pool_id": pool.id,
                        "question_text": q_data.get("stem", ""),
                        "options": json.dumps(q_data.get("options") or []),
                        "correct_answer": q_data.get("answer"),
                        "concept": tags.get("concept"),
                        "difficulty": gq.difficulty,
                        "source_chunk_id": source_chunk_id,
                        "question_json": json.dumps(q_data),
                    }
                )

            if not rows:
                continue

            if any(r["source_chunk_id"] and str(r["source_chunk_id"]) not in existing_chunk_ids for r in rows):
                raise RuntimeError("promotion_fk_validation_failed")

            insert_stmt = (
                pg_insert(PoolQuestion)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["id"])
                .returning(PoolQuestion.id)
            )
            result = session.exec(insert_stmt)
            inserted_ids = result.scalars().all()
            session.commit()

            if inserted_ids:
                questions_promoted += len(inserted_ids)
                promoted_question_ids.extend(inserted_ids)
                if prewarm_id:
                    record_prewarm_promoted_questions(prewarm_id=prewarm_id, question_ids=inserted_ids)

            if resolved_chapter_name and (
                (pool.chapter_name or "").strip() != resolved_chapter_name
                or int(pool.chapter_number or 0) != int(chapter_num)
                or (pool.chapter or "").strip() != _build_chapter_display(chapter_num, resolved_chapter_name)
            ):
                pool.chapter_name = resolved_chapter_name
                pool.chapter_number = chapter_num
                pool.chapter = _build_chapter_display(chapter_num, resolved_chapter_name)
            if pool.status != QuestionPoolStatus.ready:
                pool.status = QuestionPoolStatus.ready
            session.add(pool)
            session.commit()

            _prewarm_pool_redis(pool=pool, new_question_ids=inserted_ids)

    return {
        "status": "success",
        "questions_promoted": questions_promoted,
        "pools_created": pools_created,
        "promoted_question_ids": promoted_question_ids,
    }


def invalidate_pools_for_syllabus(*, syllabus: str, syllabus_version: str) -> dict[str, int]:
    """
    Internal optimization helper:
    invalidate pools and Redis cache for a syllabus/version change.
    """
    redis_client = get_redis_client()
    pools_invalidated = 0
    redis_keys_deleted = 0
    with get_session() as session:
        pools = session.exec(
            select(QuestionPool).where(
                QuestionPool.syllabus == syllabus,
                QuestionPool.syllabus_version == syllabus_version,
            )
        ).all()
        for pool in pools:
            pool.status = QuestionPoolStatus.pending
            pool.last_error = "invalidated_due_to_syllabus_update"
            session.add(pool)
            pools_invalidated += 1
            pool_key = get_pool_key(
                pool.syllabus,
                pool.class_level,
                pool.subject,
                pool.difficulty,
                pool.syllabus_version,
            )
            try:
                redis_keys_deleted += int(redis_client.delete(pool_key) or 0)
            except Exception:
                continue
        session.commit()
    return {"pools_invalidated": pools_invalidated, "redis_keys_deleted": redis_keys_deleted}

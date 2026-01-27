from __future__ import annotations

import json
import random
from typing import Literal

from app.cache.redis_client import get_redis_client
from app.db.models import Chunk, PoolQuestion, QuestionPool, QuestionPoolStatus
from app.db.session import get_session
from app.observability.logging import log
from sqlmodel import select


def get_pool_key(
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
    difficulty: str,
    syllabus_version: str,
) -> str:
    """Generate Redis key for a question pool."""
    return f"qp:{syllabus}:{class_level}:{subject}:{chapter}:{difficulty}:{syllabus_version}"


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
) -> QuestionPool:
    """Get existing pool or create a new one in pending status."""
    with get_session() as session:
        pool = session.exec(
            select(QuestionPool).where(
                QuestionPool.syllabus == syllabus,
                QuestionPool.class_level == class_level,
                QuestionPool.subject == subject,
                QuestionPool.chapter == chapter,
                QuestionPool.difficulty == difficulty,
                QuestionPool.syllabus_version == syllabus_version,
            )
        ).first()
        
        if not pool:
            pool = QuestionPool(
                syllabus=syllabus,
                class_level=class_level,
                subject=subject,
                chapter=chapter,
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
                key=get_pool_key(syllabus, class_level, subject, chapter, difficulty, syllabus_version),
            )
        
        return pool


def get_pool_status(
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
    difficulty: str,
    syllabus_version: str = "2024",
) -> QuestionPoolStatus | None:
    """Get pool status from database."""
    with get_session() as session:
        pool = session.exec(
            select(QuestionPool).where(
                QuestionPool.syllabus == syllabus,
                QuestionPool.class_level == class_level,
                QuestionPool.subject == subject,
                QuestionPool.chapter == chapter,
                QuestionPool.difficulty == difficulty,
                QuestionPool.syllabus_version == syllabus_version,
            )
        ).first()
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
    pool_key = get_pool_key(syllabus, class_level, subject, chapter, difficulty, syllabus_version)
    
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
            pool.chapter,
            pool.difficulty,
            pool.syllabus_version,
        )
        
        redis_client = get_redis_client()
        
        # Shuffle and add to Redis list
        shuffled_ids = question_ids.copy()
        random.shuffle(shuffled_ids)
        
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

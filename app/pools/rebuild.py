"""
Rebuild Redis pool from database questions.
Called when DB has pool but Redis is empty/missing.
"""

from __future__ import annotations

import random

from app.cache.redis_client import get_redis_client
from app.db.models import PoolQuestion, QuestionPool
from app.db.session import get_session
from app.observability.logging import log
from app.pools.service import get_pool_key
from sqlmodel import select


def rebuild_redis_pool_async(pool_id: str) -> None:
    """
    Rebuild Redis pool from database questions (async, non-blocking).
    This is called when DB has pool but Redis is empty.
    """
    try:
        with get_session() as session:
            pool = session.get(QuestionPool, pool_id)
            if not pool:
                log.warning("pool_not_found_for_rebuild", pool_id=pool_id)
                return
            
            if pool.status.value != "ready":
                log.warning("pool_not_ready_for_rebuild", pool_id=pool_id, status=pool.status.value)
                return
            
            # Fetch all question IDs from DB for this pool
            questions = session.exec(
                select(PoolQuestion.id).where(PoolQuestion.pool_id == pool_id)
            ).all()
            
            if not questions:
                log.warning("no_questions_in_pool_for_rebuild", pool_id=pool_id)
                return
            
            question_ids = [str(q) for q in questions]
            
            # Rebuild Redis list
            pool_key = get_pool_key(
                pool.syllabus,
                pool.class_level,
                pool.subject,
                pool.difficulty,
                pool.syllabus_version,
            )
            
            redis_client = get_redis_client()
            
            # Clear existing Redis key (if any)
            redis_client.delete(pool_key)
            
            # Shuffle and add to Redis
            shuffled_ids = question_ids.copy()
            random.shuffle(shuffled_ids)
            
            if shuffled_ids:
                redis_client.rpush(pool_key, *shuffled_ids)
                redis_client.expire(pool_key, 30 * 24 * 60 * 60)  # 30 days TTL
                log.info(
                    "redis_pool_rebuilt",
                    pool_id=pool_id,
                    pool_key=pool_key,
                    question_count=len(shuffled_ids),
                )
            else:
                log.warning("no_questions_to_rebuild", pool_id=pool_id)
                
    except Exception as e:
        log.error("redis_pool_rebuild_failed", pool_id=pool_id, error=str(e), error_type=type(e).__name__)

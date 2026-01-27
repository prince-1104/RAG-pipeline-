from __future__ import annotations

import concurrent.futures
import json
import uuid
from typing import Literal

from app.api.v1.schemas.generation import Blueprint, DifficultyMix, QuestionConstraints
from app.db.models import Chunk, PoolQuestion
from app.db.session import get_session
from app.generation.sync import generate_questions_sync
from app.observability.logging import log
from app.pools.service import add_questions_to_pool, get_or_create_pool
from app.settings import get_settings
from sqlmodel import select


def run_admin_generate_job(job) -> dict:
    """
    Admin-triggered question generation job.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    Generates questions based on admin criteria and stores them in:
    - Database (PoolQuestion table)
    - Redis (for fast retrieval by institutes)
    
    Steps:
    1. Parse job request (document_id, class range, subject, difficulty, count)
    2. For each class in range:
       a. Retrieve relevant chunks
       b. Generate questions using LLM
       c. Store questions in DB and Redis pools
    """
    job_data = json.loads(job.request_json)
    
    document_id = job_data["document_id"]
    corpus_id = job_data.get("corpus_id")
    from_class = job_data["from_class"]
    to_class = job_data["to_class"]
    subject = job_data["subject"]
    difficulty = job_data["difficulty"]
    question_count = job_data["question_count"]
    question_types = job_data.get("question_types", ["mcq", "short"])
    chapters = job_data.get("chapters")
    syllabus = job_data.get("syllabus", "CBSE")
    syllabus_version = job_data.get("syllabus_version", "2024")
    
    log.info(
        "admin_generate_job_started",
        job_id=job.id,
        document_id=document_id,
        from_class=from_class,
        to_class=to_class,
        subject=subject,
        difficulty=difficulty,
        question_count=question_count,
    )
    
    # Resolve corpus_id fallback for legacy jobs (admin-only)
    if not corpus_id:
        tenant_fallback = job_data.get("tenant_id") or getattr(job, "tenant_id", None)
        log.warning(
            "corpus_id missing; falling back to tenant_id (admin-only)",
            job_id=job.id,
            tenant_id=tenant_fallback,
        )
        if tenant_fallback:
            from app.db.models import Corpus

            with get_session() as session:
                corpus = session.exec(
                    select(Corpus).where(Corpus.tenant_id == tenant_fallback).limit(1)
                ).first()
            if not corpus:
                raise ValueError("corpus_not_found_for_tenant_fallback")
            corpus_id = corpus.id
        else:
            raise ValueError("corpus_id_missing_and_no_tenant_fallback")

    # Get document to verify it exists
    with get_session() as session:
        from app.db.models import Document
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
    
    total_questions_generated = 0
    pools_updated = []
    
    settings = get_settings()
    max_workers = max(1, int(getattr(settings, "admin_generate_max_concurrency", 2) or 2))
    classes_in_range = to_class - from_class + 1
    questions_per_class = max(1, question_count // classes_in_range)

    def _run_for_class(class_level: int) -> dict | None:
        try:
            # Determine chapters to use
            chapters_to_use = chapters
            if not chapters_to_use:
                # Auto-detect chapters from chunks
                with get_session() as session:
                    from app.db.models import Chunk

                    chunk_rows = session.exec(
                        select(Chunk.chapter)
                        .where(
                            Chunk.corpus_id == corpus_id,
                            Chunk.class_level == class_level,
                            Chunk.subject == subject,
                        )
                        .distinct()
                    ).all()
                    chapters_to_use = [str(c).strip() for c in chunk_rows if c and str(c).strip()]

            if not chapters_to_use:
                chapters_to_use = ["Unknown Chapter"]

            # Generate questions for this class
            questions = _generate_questions_for_class(
                corpus_id=corpus_id,
                document_id=document_id,
                class_level=class_level,
                subject=subject,
                difficulty=difficulty,
                question_types=question_types,
                chapters=chapters_to_use,
                count=questions_per_class,
            )

            if not questions:
                log.warning(
                    "no_questions_generated_for_class",
                    job_id=job.id,
                    class_level=class_level,
                )
                return None

            # Store questions in pool
            question_ids = _store_questions_in_pool(
                questions=questions,
                corpus_id=corpus_id,
                class_level=class_level,
                subject=subject,
                syllabus=syllabus,
                syllabus_version=syllabus_version,
                difficulty=difficulty,
            )

            # Add to Redis
            add_questions_to_pool(pool.id, question_ids)

            log.info(
                "admin_questions_generated_for_class",
                job_id=job.id,
                class_level=class_level,
                question_count=len(question_ids),
                pool_id=pool.id,
            )

            return {
                "pool_id": pool.id,
                "class_level": class_level,
                "question_count": len(question_ids),
            }
        except Exception as e:  # noqa: BLE001
            log.exception(
                "admin_generate_failed_for_class",
                job_id=job.id,
                class_level=class_level,
                error=str(e),
            )
            return None

    class_levels = list(range(from_class, to_class + 1))
    log.info(
        "admin_generate_parallel_start",
        job_id=job.id,
        classes=class_levels,
        questions_per_class=questions_per_class,
        max_workers=max_workers,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_for_class, cl): cl for cl in class_levels}
        for future in concurrent.futures.as_completed(futures.keys()):
            res = future.result()
            if not res:
                continue
            pools_updated.append(res)
            total_questions_generated += int(res["question_count"])
    
    result = {
        "ok": True,
        "total_questions_generated": total_questions_generated,
        "pools_updated": pools_updated,
        "document_id": document_id,
        "from_class": from_class,
        "to_class": to_class,
        "subject": subject,
        "difficulty": difficulty,
    }
    
    log.info(
        "admin_generate_job_completed",
        job_id=job.id,
        total_questions=total_questions_generated,
        pools_count=len(pools_updated),
    )
    
    return result


def _generate_questions_for_class(
    corpus_id: str,
    document_id: str,
    class_level: int,
    subject: str,
    difficulty: Literal["easy", "medium", "hard"],
    question_types: list[str],
    chapters: list[str],
    count: int,
) -> list[dict]:
    """
    Generate questions for a specific class using the existing generation pipeline.
    Reuses the same logic as sync generation but stores results differently.
    """
    settings = get_settings()
    
    # Build blueprint
    difficulty_mix = DifficultyMix(**{difficulty: count, "easy": 0, "medium": 0, "hard": 0})
    
    blueprint = Blueprint(
        blueprint_version="2026-01-02",
        job_summary={
            "skills": ["problem-solving", "conceptual understanding"],
            "role": "student assessment",
            "focus_topics": [subject.lower()],
        },
        constraints=QuestionConstraints(
            class_level=class_level,
            subject=subject,
            chapters=chapters,
            language="en",
        ),
        question_types=question_types,
        count=count,
        difficulty_mix=difficulty_mix,
        coverage_strategy="balanced",
    )
    
    # Use existing sync generation logic
    from app.api.v1.schemas.generation import GenerateRequest
    
    generate_req = GenerateRequest(
        tenant_id="admin",
        corpus_id=corpus_id,
        blueprint=blueprint,
        max_context_chunks=settings.max_context_chunks,
    )
    
    # Generate questions synchronously (this is in worker, so it's OK)
    response = generate_questions_sync(generate_req)
    
    if not response.questions:
        return []
    
    # Convert to dict format for storage
    questions = []
    for q in response.questions:
        q_dict = {
            "stem": q.stem,
            "qtype": q.qtype,
            "difficulty": q.difficulty,
            "language": q.language,
            "options": q.options or [],
            "answer": q.answer or "",
            "explanation": q.explanation or "",
            "tags": q.tags or {},
            "citations": [c.model_dump() for c in q.citations],
        }
        questions.append(q_dict)
    
    return questions


def _store_questions_in_pool(
    questions: list[dict],
    corpus_id: str,
    class_level: int,
    subject: str,
    syllabus: str,
    syllabus_version: str,
    difficulty: str,
) -> list[str]:
    """Store questions in chapter pools and return question IDs."""
    question_ids: list[str] = []

    chunk_ids: set[str] = set()
    for q in questions:
        for cit in q.get("citations", []) or []:
            cid = cit.get("chunk_id")
            if cid:
                chunk_ids.add(str(cid))

    chunk_chapters: dict[str, str] = {}
    if chunk_ids:
        with get_session() as session:
            rows = session.exec(
                select(Chunk.id, Chunk.chapter).where(Chunk.id.in_(list(chunk_ids)))
            ).all()
        chunk_chapters = {
            str(r[0]): (str(r[1]).strip() if r[1] else "Unknown Chapter")
            for r in rows
        }

    questions_by_chapter: dict[str, list[dict]] = {}
    for q in questions:
        chapter = (q.get("tags") or {}).get("chapter")
        if not chapter:
            cid = (q.get("citations") or [{}])[0].get("chunk_id")
            chapter = chunk_chapters.get(str(cid), None)
        chapter = str(chapter).strip() if chapter else "Unknown Chapter"
        questions_by_chapter.setdefault(chapter, []).append(q)

    with get_session() as session:
        for chapter, chapter_questions in questions_by_chapter.items():
            pool = get_or_create_pool(syllabus, class_level, subject, chapter, difficulty, syllabus_version)
            for q in chapter_questions:
                q_id = str(uuid.uuid4())
                question_ids.append(q_id)

                # Extract fields
                stem = q.get("stem", "")
                options = q.get("options") or []
                answer = q.get("answer")
                citations = q.get("citations", [])
                source_chunk_id = citations[0].get("chunk_id") if citations else None
                tags = q.get("tags", {})
                concept = tags.get("concept")

                pool_question = PoolQuestion(
                    id=q_id,
                    pool_id=pool.id,
                    question_text=stem,
                    options=json.dumps(options),
                    correct_answer=answer,
                    concept=concept,
                    difficulty=difficulty,
                    source_chunk_id=source_chunk_id,
                    question_json=json.dumps(q),
                )
                session.add(pool_question)

        session.commit()

    log.info("admin_questions_stored_in_pool", count=len(question_ids))
    return question_ids

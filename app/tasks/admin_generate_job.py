from __future__ import annotations

import asyncio
import json
import re
from typing import Literal

from app.api.v1.schemas.generation import Blueprint, DifficultyMix, QuestionConstraints
from app.db.models import Chunk, GeneratedQuestion, QuestionCitation
from app.db.session import get_session_ctx
from app.generation.sync import generate_questions_sync
from app.observability.logging import log
from app.settings import get_settings
from sqlalchemy import select


async def run_admin_generate_job(job) -> dict:
    """
    Admin-triggered question generation job.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    Generates questions based on admin criteria and stores them in:
    - Database (GeneratedQuestion table + QuestionCitation)
    
    Steps:
    1. Parse job request (document_id, class range, subject, difficulty, count)
    2. For each class in range:
       a. Retrieve relevant chunks
       b. Generate questions using LLM
       c. Store questions in DB
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

            async with get_session_ctx() as session:
                corpus = (await session.execute(
                    select(Corpus).where(Corpus.tenant_id == tenant_fallback).limit(1)
                )).scalars().first()
            if not corpus:
                raise ValueError("corpus_not_found_for_tenant_fallback")
            corpus_id = corpus.id
        else:
            raise ValueError("corpus_id_missing_and_no_tenant_fallback")

    # Get document to verify it exists
    async with get_session_ctx() as session:
        from app.db.models import Document
        document = await session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
    
    total_questions_generated = 0
    pools_updated = []
    
    settings = get_settings()
    max_workers = max(1, int(getattr(settings, "admin_generate_max_concurrency", 2) or 2))
    classes_in_range = to_class - from_class + 1
    questions_per_class = max(1, question_count // classes_in_range)

    async def _run_for_class(class_level: int) -> dict | None:
        try:
            # Determine chapters to use
            chapters_to_use = chapters
            if not chapters_to_use:
                # Auto-detect chapters from chunks
                async with get_session_ctx() as session:
                    from app.db.models import Chunk

                    chunk_rows = (await session.execute(
                        select(Chunk.chapter)
                        .where(
                            Chunk.corpus_id == corpus_id,
                            Chunk.class_level == class_level,
                            Chunk.subject == subject,
                        )
                        .distinct()
                    )).scalars().all()
                    chapters_to_use = [str(c).strip() for c in chunk_rows if c and str(c).strip()]

            if not chapters_to_use:
                chapters_to_use = ["Unknown Chapter"]

            # Generate questions for this class
            questions = await _generate_questions_for_class(
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

            question_ids = await _store_generated_questions(
                questions=questions,
                job_id=job.id,
                corpus_id=corpus_id,
                class_level=class_level,
                subject=subject,
                difficulty=difficulty,
                chapter_fallback=chapters_to_use[0] if chapters_to_use else "Unknown Chapter",
            )

            log.info(
                "admin_questions_generated_for_class",
                job_id=job.id,
                class_level=class_level,
                question_count=len(question_ids),
            )

            return {
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

    results = await asyncio.gather(*[_run_for_class(cl) for cl in class_levels])
    for res in results:
        if not res:
            continue
        pools_updated.append(res)
        total_questions_generated += int(res["question_count"])
    
    result = {
        "ok": True,
        "total_questions_generated": total_questions_generated,
        "classes_updated": pools_updated,
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
        classes_count=len(pools_updated),
    )
    
    return result


async def _generate_questions_for_class(
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

    # Split total count across chapters (per-chapter generation).
    chapters = [c for c in chapters if c and str(c).strip()]
    if not chapters:
        chapters = ["Unknown Chapter"]

    base = max(0, count // len(chapters)) if chapters else 0
    remainder = count - (base * len(chapters))
    chapter_counts: dict[str, int] = {}
    for idx, chapter in enumerate(chapters):
        extra = 1 if idx < remainder else 0
        chapter_counts[chapter] = base + extra

    async def _generate_for_chapter(chapter: str, chapter_count: int) -> list[dict]:
        if chapter_count <= 0:
            return []

        difficulty_mix = DifficultyMix(**{difficulty: chapter_count, "easy": 0, "medium": 0, "hard": 0})
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
                chapters=[chapter],
                language="en",
            ),
            question_types=question_types,
            count=chapter_count,
            difficulty_mix=difficulty_mix,
            coverage_strategy="balanced",
        )

        from app.api.v1.schemas.generation import GenerateRequest

        generate_req = GenerateRequest(
            tenant_id="admin",
            corpus_id=corpus_id,
            blueprint=blueprint,
            max_context_chunks=settings.max_context_chunks,
        )

        response = await generate_questions_sync(generate_req)
        if not response.questions:
            return []

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

    questions_all: list[dict] = []
    chapter_workers = max(
        1,
        min(
            int(getattr(settings, "admin_generate_chapter_concurrency", 1) or 1),
            len(chapters),
        ),
    )

    if chapter_workers == 1 or len(chapters) == 1:
        for chapter, chapter_count in chapter_counts.items():
            try:
                questions_all.extend(await _generate_for_chapter(chapter, chapter_count))
            except Exception as e:  # noqa: BLE001
                log.exception(
                    "admin_generate_failed_for_chapter",
                    class_level=class_level,
                    chapter=chapter,
                    error=str(e),
                )
    else:
        chapter_items = list(chapter_counts.items())
        results = await asyncio.gather(
            *[_generate_for_chapter(chapter, chapter_count) for chapter, chapter_count in chapter_items],
            return_exceptions=True,
        )
        for (chapter, _), result in zip(chapter_items, results):
            if isinstance(result, BaseException):
                log.exception(
                    "admin_generate_failed_for_chapter",
                    class_level=class_level,
                    chapter=chapter,
                    error=str(result),
                )
            else:
                questions_all.extend(result)

    return questions_all


async def _store_generated_questions(
    questions: list[dict],
    job_id: str,
    corpus_id: str,
    class_level: int,
    subject: str,
    difficulty: str,
    chapter_fallback: str,
) -> list[str]:
    """Store questions in GeneratedQuestion + QuestionCitation tables."""
    question_ids: list[str] = []

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

    citation_chunk_ids = {
        cit.get("chunk_id")
        for q in questions
        for cit in (q.get("citations") or [])
        if cit.get("chunk_id")
    }
    async with get_session_ctx() as session:
        valid_chunk_query = select(Chunk.id).where(
            Chunk.corpus_id == corpus_id,
            Chunk.id.in_(citation_chunk_ids),
        ) if citation_chunk_ids else None
        valid_chunk_ids = set(
            (await session.execute(valid_chunk_query)).scalars().all() if valid_chunk_query else []
        )

        for q in questions:
            q_data = dict(q)
            tags = q_data.get("tags") or {}
            if not tags.get("chapter"):
                tags["chapter"] = chapter_fallback
            q_data["tags"] = tags
            chapter_value = str(tags.get("chapter") or chapter_fallback)
            chapter_number, chapter_name = _parse_chapter_identity(chapter_value, chapter_fallback)

            dq = GeneratedQuestion(
                job_id=job_id,
                corpus_id=corpus_id,
                class_level=class_level,
                subject=subject,
                chapter=chapter_value,
                chapter_number=chapter_number,
                chapter_name=chapter_name,
                difficulty=difficulty,
                qtype=str(q_data.get("qtype") or ""),
                language=str(q_data.get("language") or "en"),
                question_json=json.dumps(q_data),
                dedupe_key=str(tags.get("dedupe_key") or ""),
            )
            session.add(dq)
            await session.commit()
            await session.refresh(dq)
            question_ids.append(dq.id)

            for cit in q_data.get("citations") or []:
                chunk_id = cit.get("chunk_id")
                if chunk_id not in valid_chunk_ids:
                    continue
                session.add(
                    QuestionCitation(
                        question_id=dq.id,
                        chunk_id=chunk_id,
                        document_id=cit.get("document_id"),
                        storage_uri=cit.get("storage_uri"),
                        page_start=cit.get("page_start"),
                        page_end=cit.get("page_end"),
                        excerpt=cit.get("excerpt"),
                    )
                )
            await session.commit()

    log.info("admin_questions_stored_generated", count=len(question_ids))
    return question_ids

from __future__ import annotations

import json
import uuid
from typing import Literal

from app.api.v1.schemas.generation import Blueprint, DifficultyMix, QuestionConstraints
from app.db.models import PoolQuestion
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
    corpus_id = job_data["corpus_id"]
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
    
    # Get document to verify it exists
    with get_session() as session:
        from app.db.models import Document
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
    
    total_questions_generated = 0
    pools_updated = []
    
    # Generate questions for each class in the range
    for class_level in range(from_class, to_class + 1):
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
            
            # Calculate questions per class (distribute evenly)
            classes_in_range = to_class - from_class + 1
            questions_per_class = max(1, question_count // classes_in_range)
            
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
                continue
            
            # Store questions in pool
            pool = get_or_create_pool(syllabus, class_level, subject, difficulty, syllabus_version)
            question_ids = _store_questions_in_pool(
                pool_id=pool.id,
                questions=questions,
                corpus_id=corpus_id,
                class_level=class_level,
                subject=subject,
            )
            
            # Add to Redis
            add_questions_to_pool(pool.id, question_ids)
            pools_updated.append({
                "pool_id": pool.id,
                "class_level": class_level,
                "question_count": len(question_ids),
            })
            
            total_questions_generated += len(question_ids)
            
            log.info(
                "admin_questions_generated_for_class",
                job_id=job.id,
                class_level=class_level,
                question_count=len(question_ids),
                pool_id=pool.id,
            )
            
        except Exception as e:
            log.exception(
                "admin_generate_failed_for_class",
                job_id=job.id,
                class_level=class_level,
                error=str(e),
            )
            # Continue with other classes
            continue
    
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
    pool_id: str,
    questions: list[dict],
    corpus_id: str,
    class_level: int,
    subject: str,
) -> list[str]:
    """Store questions in PoolQuestion table and return question IDs."""
    question_ids = []
    
    with get_session() as session:
        for q in questions:
            q_id = str(uuid.uuid4())
            question_ids.append(q_id)
            
            # Extract fields
            stem = q.get("stem", "")
            options = q.get("options") or []
            answer = q.get("answer")
            difficulty = q.get("difficulty", "medium")
            citations = q.get("citations", [])
            source_chunk_id = citations[0].get("chunk_id") if citations else None
            tags = q.get("tags", {})
            concept = tags.get("concept")
            
            pool_question = PoolQuestion(
                id=q_id,
                pool_id=pool_id,
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
    
    log.info("admin_questions_stored_in_pool", pool_id=pool_id, count=len(question_ids))
    return question_ids

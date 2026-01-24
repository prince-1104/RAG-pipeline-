from __future__ import annotations

import json
import uuid
from typing import Literal

from app.db.models import (
    Chunk,
    Document,
    PoolQuestion,
    QuestionPool,
    QuestionPoolStatus,
)
from app.db.session import get_session
from app.generation.guardrails import (
    dedupe_key_for,
    reject_if_embedding_duplicate,
    reject_if_exact_duplicate,
    reject_if_low_support,
)
from app.generation.prompts import build_system_prompt
from app.generation.schema import QUESTION_SET_SCHEMA
from app.generation.validation import ensure_mandatory_citations, validate_strict_schema
from app.llm.factory import get_llm_client
from app.observability.logging import log
from app.pools.service import (
    add_questions_to_pool,
    get_or_create_pool,
    mark_pool_failed,
    mark_pool_generating,
)
from app.retrieval.retriever import retrieve
from sqlmodel import select


def generate_question_pool(
    corpus_id: str,
    syllabus: str,
    class_level: int,
    subject: str,
    difficulty: Literal["easy", "medium", "hard"],
    syllabus_version: str = "2024",
    target_count: int = 200,  # Generate 200 questions per pool
) -> None:
    """
    Background worker: Generate a question pool for a specific syllabus/class/subject/difficulty.
    
    Steps:
    1. Check if pool already exists and is ready → exit
    2. Retrieve TOP 5-8 chunks from Qdrant
    3. Phase 1: Extract examinable concepts
    4. Phase 2: Generate questions per concept (parallel where possible)
    5. Store questions in DB
    6. Push question_ids into Redis
    7. Mark pool as READY
    """
    # Step 1: Get or create pool
    pool = get_or_create_pool(syllabus, class_level, subject, difficulty, syllabus_version)
    
    if pool.status == QuestionPoolStatus.ready:
        log.info("pool_already_ready", pool_id=pool.id)
        return
    
    # Mark as generating
    mark_pool_generating(pool.id)
    
    try:
        # Step 2: Retrieve top 5-8 chunks (STRICT LIMIT)
        top_k = 8
        query = f"{syllabus} {subject} class {class_level} syllabus version {syllabus_version}"
        
        # Use a generic query that retrieves diverse content
        rr = retrieve(
            corpus_id=corpus_id,
            class_level=class_level,
            subject=subject,
            chapters=["Unknown Chapter"],  # Get all chapters for pool generation
            query=query,
            top_k=top_k,
        )
        
        if not rr.chunks or len(rr.chunks) < 3:
            raise ValueError(f"Insufficient chunks retrieved: {len(rr.chunks) if rr.chunks else 0}")
        
        log.info("chunks_retrieved_for_pool", pool_id=pool.id, chunk_count=len(rr.chunks))
        
        # Get document URIs
        with get_session() as session:
            docs = session.exec(select(Document).where(Document.corpus_id == corpus_id)).all()
        doc_uri_map = {d.id: d.storage_uri for d in docs}
        
        # Step 3: Phase 1 - Extract examinable concepts
        concepts = _extract_concepts(rr, doc_uri_map, difficulty)
        
        if not concepts:
            raise ValueError("No examinable concepts extracted")
        
        log.info("concepts_extracted", pool_id=pool.id, concept_count=len(concepts))
        
        # Step 4: Phase 2 - Generate questions per concept
        all_questions: list[dict] = []
        questions_per_concept = max(1, target_count // len(concepts))
        
        for concept in concepts:
            try:
                concept_questions = _generate_questions_for_concept(
                    concept=concept,
                    retrieval=rr,
                    doc_uri_map=doc_uri_map,
                    difficulty=difficulty,
                    count=questions_per_concept,
                    corpus_id=corpus_id,
                )
                all_questions.extend(concept_questions)
                
                # Stop if we have enough questions
                if len(all_questions) >= target_count:
                    break
            except Exception as e:
                log.warning("concept_generation_failed", concept=concept, error=str(e))
                continue
        
        if len(all_questions) < 50:  # Minimum viable pool size
            raise ValueError(f"Insufficient questions generated: {len(all_questions)}")
        
        # Step 5: Store questions in DB
        question_ids = _store_pool_questions(pool.id, all_questions, corpus_id, class_level, subject)
        
        # Step 6: Add to Redis
        add_questions_to_pool(pool.id, question_ids)
        
        log.info("pool_generation_complete", pool_id=pool.id, question_count=len(question_ids))
        
    except Exception as e:
        log.exception("pool_generation_error", pool_id=pool.id, error=str(e))
        mark_pool_failed(pool.id, str(e))
        raise


def _extract_concepts(
    retrieval_result,
    doc_uri_map: dict[str, str],
    difficulty: str,
) -> list[str]:
    """
    Phase 1: Ask LLM to list examinable concepts only.
    Returns list of concept names.
    """
    client = get_llm_client()
    system_prompt = (
        "You are an educational content analyzer for CBSE.\n"
        "Your task: Extract examinable concepts from the provided context.\n"
        "Return ONLY a JSON array of concept names (strings).\n"
        "Example: [\"Photosynthesis\", \"Cell Structure\", \"Chemical Reactions\"]\n"
        f"Focus on concepts suitable for {difficulty} difficulty questions."
    )
    
    # Build context from retrieval
    ctx_lines: list[str] = []
    for r in retrieval_result.chunks[:5]:  # Use top 5 for concept extraction
        c = r.chunk
        uri = doc_uri_map.get(c.document_id, "unknown")
        chunk_text = c.text[:1000]  # Shorter for concept extraction
        ctx_lines.extend([
            f"CHAPTER: {c.chapter}",
            f"TEXT: {chunk_text[:500]}",
            "---",
        ])
    
    user_prompt = "\n".join([
        "CONTEXT:",
        "\n".join(ctx_lines),
        "",
        f"Extract 8-12 examinable concepts suitable for {difficulty} difficulty questions.",
        "Return JSON array of concept names only.",
    ])
    
    try:
        # Use a simpler schema for concept extraction
        concept_schema = {
            "type": "object",
            "properties": {
                "concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["concepts"],
        }
        
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=concept_schema,
        )
        
        concepts = result.parsed_json.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []
        
        # Filter and normalize
        concepts = [c.strip() for c in concepts if isinstance(c, str) and c.strip()][:12]
        
        return concepts if concepts else ["General Concepts"]  # Fallback
        
    except Exception as e:
        log.warning("concept_extraction_failed", error=str(e))
        # Fallback: use chapter names as concepts
        return list({r.chunk.chapter for r in retrieval_result.chunks[:5] if r.chunk.chapter})


def _generate_questions_for_concept(
    concept: str,
    retrieval,
    doc_uri_map: dict[str, str],
    difficulty: str,
    count: int,
    corpus_id: str,
) -> list[dict]:
    """
    Phase 2: Generate N questions for a specific concept.
    LLM prompt MUST specify difficulty explicitly.
    """
    client = get_llm_client()
    system_prompt = build_system_prompt()
    
    # Build context with concept focus
    ctx_lines: list[str] = []
    for r in retrieval.chunks[:8]:
        c = r.chunk
        uri = doc_uri_map.get(c.document_id, "unknown")
        chunk_text = c.text[:1500]
        excerpt = c.text[:200].replace("\n", " ")
        ctx_lines.extend([
            "----",
            f"CHUNK_ID: {c.id}",
            f"DOCUMENT_ID: {c.document_id}",
            f"STORAGE_URI: {uri}",
            f"PAGES: {c.page_start}-{c.page_end}",
            f"CHAPTER: {c.chapter}",
            f"EXCERPT: {excerpt}",
            f"TEXT: {chunk_text}",
        ])
    
    # Build difficulty-specific prompt
    difficulty_guidance = {
        "easy": "Direct factual recall, single-step reasoning, definitions, basic concepts",
        "medium": "Conceptual understanding, 1-2 step reasoning, application of concepts",
        "hard": "Multi-step reasoning, complex application, analysis, synthesis",
    }
    
    user_prompt = "\n".join([
        f"CONCEPT: {concept}",
        f"DIFFICULTY: {difficulty}",
        f"DIFFICULTY_GUIDANCE: {difficulty_guidance[difficulty]}",
        "",
        f"Generate {count} questions about '{concept}'.",
        f"ALL questions must be {difficulty} difficulty.",
        "Question types: mcq, short, long (mix as appropriate).",
        "",
        "CONTEXT:",
        "\n".join(ctx_lines),
        "",
        "Return JSON with 'questions' array and 'rejected' array.",
        "Each question must have citations pointing to relevant chunks.",
    ])
    
    try:
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=QUESTION_SET_SCHEMA,
        )
        
        # Normalize output
        from app.generation.sync import _normalize_llm_output
        _normalize_llm_output(result.parsed_json)
        
        # Validate
        issues = validate_strict_schema(result.parsed_json) + ensure_mandatory_citations(result.parsed_json)
        if issues:
            log.warning("question_validation_issues", concept=concept, issues_count=len(issues))
            # Continue with partial results
        
        questions = result.parsed_json.get("questions", []) or []
        
        # Fix citations
        for q in questions:
            for cit in q.get("citations", []) or []:
                did = cit.get("document_id")
                cit["storage_uri"] = doc_uri_map.get(did, cit.get("storage_uri", ""))
        
        # Filter by difficulty and apply guardrails
        valid_questions = []
        for q in questions:
            # Ensure difficulty matches
            q["difficulty"] = difficulty
            
            # Basic guardrails
            stem = str(q.get("stem") or "")
            qtype = str(q.get("qtype") or "")
            chapter = str((q.get("tags") or {}).get("chapter") or retrieval.chunks[0].chunk.chapter if retrieval.chunks else "Unknown")
            dk = dedupe_key_for(stem, qtype, chapter)
            
            rej = reject_if_exact_duplicate(corpus_id=corpus_id, key=dk) or reject_if_embedding_duplicate(
                corpus_id=corpus_id, stem=stem
            )
            if not rej:
                c0 = (q.get("citations") or [{}])[0]
                rej = reject_if_low_support(
                    answer=q.get("answer"),
                    explanation=q.get("explanation"),
                    excerpt=str(c0.get("excerpt") or ""),
                )
            
            if not rej:
                q["tags"] = {**(q.get("tags") or {}), "chapter": chapter, "dedupe_key": dk, "concept": concept}
                valid_questions.append(q)
        
        return valid_questions[:count]  # Limit to requested count
        
    except Exception as e:
        log.warning("question_generation_failed", concept=concept, error=str(e))
        return []


def _store_pool_questions(
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
            concept = q.get("tags", {}).get("concept")
            difficulty = q.get("difficulty", "medium")
            citations = q.get("citations", [])
            source_chunk_id = citations[0].get("chunk_id") if citations else None
            
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
    
    log.info("pool_questions_stored", pool_id=pool_id, count=len(question_ids))
    return question_ids

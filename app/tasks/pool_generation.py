from __future__ import annotations

import concurrent.futures
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
    mark_pool_ready_partial,
)
from app.retrieval.retriever import retrieve
from app.settings import get_settings
from sqlmodel import select


def _pool_system_prompt(difficulty: str) -> str:
    """
    Pool generation prompt: less strict than admin/sync path.
    Goal: generate *some* valid questions quickly, even if context is imperfect.
    """
    difficulty_guidance = {
        "easy": "Focus on direct recall, definitions, and single-step questions.",
        "medium": "Focus on conceptual understanding and 1-2 step reasoning.",
        "hard": "Focus on multi-step reasoning, analysis, and synthesis.",
    }
    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])
    return (
        "You are an exam question writer for CBSE.\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT chunks. Do not invent facts.\n"
        "- If context is insufficient to generate all requested questions, generate fewer and add entries to 'rejected'.\n"
        "- Every question MUST include >=1 citation pointing to a CHUNK_ID and include a verbatim excerpt.\n"
        "- Output MUST be valid JSON with keys: questions (array), rejected (array).\n"
        "- For MCQ questions: provide exactly 4 options as SIMPLE STRINGS.\n"
        f"- Keep questions concise and aligned to the requested difficulty.\n- {guidance}\n"
    )


def _concept_chunk_id_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "chunk_id"],
                    "properties": {
                        "name": {"type": "string"},
                        "chunk_id": {"type": "string"},
                    },
                },
            }
        },
        "required": ["concepts"],
    }


def _rank_chunks_for_concept(retrieval, concept: str) -> list[Chunk]:
    """
    Heuristic: prefer chunks whose text/excerpt mention concept tokens.
    Falls back to original retrieval order.
    """
    tokens = [t for t in str(concept).lower().split() if len(t) >= 4]
    scored: list[tuple[int, int, Chunk]] = []
    for i, r in enumerate(retrieval.chunks or []):
        c = r.chunk
        hay = (c.chapter or "") + "\n" + (c.text or "")
        hay_l = hay.lower()
        hits = sum(1 for t in tokens if t in hay_l)
        scored.append((-hits, i, c))  # more hits first, then original order
    scored.sort()
    return [c for _, __, c in scored]


def _ensure_question_minimums(
    *,
    q: dict,
    concept: str,
    difficulty: str,
    chapter_fallback: str,
    citation_chunk: Chunk,
    doc_uri_map: dict[str, str],
) -> None:
    # id
    if not isinstance(q.get("id"), str) or not q.get("id"):
        q["id"] = str(uuid.uuid4())

    # language
    if not isinstance(q.get("language"), str) or q.get("language") not in ("en", "hi"):
        q["language"] = "en"

    # tags
    tags = q.get("tags")
    if not isinstance(tags, dict):
        tags = {}
    tags.setdefault("concept", concept)
    tags.setdefault("chapter", chapter_fallback or (citation_chunk.chapter or "Unknown"))
    if isinstance(q.get("qtype"), str) and q.get("qtype"):
        tags.setdefault("question_type", str(q.get("qtype")))
    q["tags"] = tags

    # difficulty
    q["difficulty"] = difficulty

    # qtype sanity
    qt = q.get("qtype")
    if not isinstance(qt, str):
        qt = ""
    qt = qt.strip().lower()
    if qt not in ("mcq", "short", "long", "case_based", "assertion_reason"):
        # infer best-effort
        qt = "mcq" if q.get("options") else "short"
    q["qtype"] = qt

    # options sanity: if mcq but options invalid, downgrade to short
    if q["qtype"] == "mcq":
        opts = q.get("options")
        if not isinstance(opts, list) or len([o for o in opts if isinstance(o, str) and o.strip()]) != 4:
            q["qtype"] = "short"
            q["options"] = None
    else:
        if "options" not in q:
            q["options"] = None

    # citations (auto-fill if missing or malformed)
    cits = q.get("citations")
    if not isinstance(cits, list) or len(cits) < 1 or not isinstance(cits[0], dict):
        cits = []
    if not cits:
        uri = doc_uri_map.get(citation_chunk.document_id, "unknown")
        excerpt = (citation_chunk.text or "").replace("\n", " ").strip()[:200]
        if len(excerpt) < 10:
            excerpt = (excerpt + " " + (citation_chunk.chapter or "context")).strip()
        q["citations"] = [
            {
                "chunk_id": str(citation_chunk.id),
                "document_id": str(citation_chunk.document_id),
                "storage_uri": str(uri or ""),
                "page_start": int(citation_chunk.page_start or 1),
                "page_end": int(citation_chunk.page_end or max(1, int(citation_chunk.page_start or 1))),
                "excerpt": excerpt[:300],
            }
        ]


def generate_question_pool(
    corpus_id: str,
    syllabus: str,
    class_level: int,
    subject: str,
    chapter: str,
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
    pool = get_or_create_pool(syllabus, class_level, subject, chapter, difficulty, syllabus_version)
    # Do not generate beyond requested_count (if already satisfied)
    if pool.requested_count is not None and pool.generated_count is not None:
        if int(pool.generated_count) >= int(pool.requested_count):
            log.info(
                "pool_generation_cap_skip",
                pool_id=pool.id,
                requested=pool.requested_count,
                generated=pool.generated_count,
            )
            return
    
    # If fully ready, no work to do.
    if pool.status == QuestionPoolStatus.ready:
        log.info("pool_already_ready", pool_id=pool.id)
        return
    
    # Mark as generating only if not already serving from a partial pool.
    if pool.status != QuestionPoolStatus.ready_partial:
        mark_pool_generating(pool.id)
    
    try:
        s = get_settings()
        requested_count = target_count
        # Difficulty-specific acceptance thresholds (fallback to global defaults)
        min_abs_by_diff = {
            "easy": int(getattr(s, "pool_generation_min_absolute_easy", 10) or 10),
            "medium": int(getattr(s, "pool_generation_min_absolute_medium", 10) or 10),
            "hard": int(getattr(s, "pool_generation_min_absolute_hard", 10) or 10),
        }
        min_frac_by_diff = {
            "easy": float(getattr(s, "pool_generation_min_fraction_easy", 0.2) or 0.2),
            "medium": float(getattr(s, "pool_generation_min_fraction_medium", 0.2) or 0.2),
            "hard": float(getattr(s, "pool_generation_min_fraction_hard", 0.2) or 0.2),
        }
        min_abs = max(1, min_abs_by_diff.get(difficulty, 10))
        min_frac = min_frac_by_diff.get(difficulty, 0.2)
        min_acceptable = max(min_abs, int(requested_count * min_frac))

        # Step 2: Retrieve top 5-8 chunks (STRICT LIMIT)
        top_k = max(3, int(getattr(s, "pool_generation_context_chunks", 6) or 6))
        query = f"{syllabus} {subject} class {class_level} chapter {chapter} syllabus version {syllabus_version}"
        
        # Use a generic query that retrieves diverse content
        rr = retrieve(
            corpus_id=corpus_id,
            class_level=class_level,
            subject=subject,
            chapters=[chapter],
            query=query,
            top_k=top_k,
            board=syllabus,
            syllabus_version=syllabus_version,
        )
        
        if not rr.chunks or len(rr.chunks) < 3:
            raise ValueError(f"Insufficient chunks retrieved: {len(rr.chunks) if rr.chunks else 0}")
        
        log.info("chunks_retrieved_for_pool", pool_id=pool.id, chunk_count=len(rr.chunks))
        
        # Get document URIs
        with get_session() as session:
            docs = session.exec(select(Document).where(Document.corpus_id == corpus_id)).all()
        doc_uri_map = {d.id: d.storage_uri for d in docs}
        
        # Step 3: Phase 1 - Extract examinable concepts (+ supporting chunk_id)
        extracted = _extract_concepts(rr, doc_uri_map, difficulty)
        concepts = [c["name"] for c in extracted]
        concept_to_chunk_id = {c["name"]: c.get("chunk_id", "") for c in extracted}
        
        if not concepts:
            raise ValueError("No examinable concepts extracted")
        
        log.info("concepts_extracted", pool_id=pool.id, concept_count=len(concepts))
        
        # Step 4: Phase 2 - Generate questions per concept
        all_questions: list[dict] = []
        questions_per_concept = min(8, max(1, target_count // len(concepts)))

        max_workers = max(1, int(getattr(s, "pool_generation_max_concurrency", 6) or 6))
        log.info(
            "pool_generation_parallel_start",
            pool_id=pool.id,
            concepts=len(concepts),
            questions_per_concept=questions_per_concept,
            max_workers=max_workers,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept: dict[concurrent.futures.Future[list[dict]], str] = {}
            for concept in concepts:
                future = executor.submit(
                    _generate_questions_for_concept,
                    concept=concept,
                    retrieval=rr,
                    doc_uri_map=doc_uri_map,
                    difficulty=difficulty,
                    count=questions_per_concept,
                    corpus_id=corpus_id,
                    preferred_chunk_id=concept_to_chunk_id.get(concept) or None,
                )
                future_to_concept[future] = concept

            for future in concurrent.futures.as_completed(future_to_concept.keys()):
                concept = future_to_concept[future]
                try:
                    concept_questions = future.result()
                    all_questions.extend(concept_questions)
                except Exception as e:  # noqa: BLE001
                    log.warning("concept_generation_failed", concept=concept, error=str(e)[:200])
                    continue

                if len(all_questions) >= target_count:
                    # Best-effort cancellation of remaining work
                    for f in future_to_concept.keys():
                        if f is not future:
                            f.cancel()
                    break

        generated = len(all_questions)
        log.info(
            "pool_generation_yield",
            pool_id=pool.id,
            requested=requested_count,
            generated=generated,
            min_acceptable=min_acceptable,
        )

        if generated == 0:
            # Hard fail only when absolutely nothing was generated.
            mark_pool_failed(pool.id, "insufficient_questions:0")
            return

        # Step 5: Store questions in DB
        question_ids = _store_pool_questions(pool.id, all_questions, corpus_id, class_level, subject)

        # Step 6: Add to Redis and set pool status
        if generated < requested_count:
            # Best-effort: partial pool is still serveable; trigger backfill via API path if needed.
            add_questions_to_pool(
                pool.id,
                question_ids,
                requested_count=requested_count,
                generated_count=generated,
                status=QuestionPoolStatus.ready_partial,
                last_error=("insufficient_questions" if generated < min_acceptable else None),
            )
            # Persist status/counters even if Redis add fails
            mark_pool_ready_partial(
                pool.id,
                requested_count=requested_count,
                generated_count=generated,
                last_error=("insufficient_questions" if generated < min_acceptable else None),
            )
        else:
            add_questions_to_pool(
                pool.id,
                question_ids,
                requested_count=requested_count,
                generated_count=generated,
                status=QuestionPoolStatus.ready,
                last_error=None,
            )
        
        log.info("pool_generation_complete", pool_id=pool.id, question_count=len(question_ids))
        
    except Exception as e:
        # Do not crash the worker; mark failed and exit.
        log.exception("pool_generation_error", pool_id=pool.id, error=str(e))
        mark_pool_failed(pool.id, str(e)[:500])
        return


def _extract_concepts(
    retrieval_result,
    doc_uri_map: dict[str, str],
    difficulty: str,
) -> list[dict]:
    """
    Phase 1: Ask LLM to list examinable concepts grounded in context.
    Returns list of dicts: {"name": str, "chunk_id": str}
    """
    s = get_settings()
    client = get_llm_client()
    system_prompt = (
        "You are an educational content analyzer for CBSE.\n"
        "Your task: Extract examinable concepts from the provided context.\n"
        "Return concepts grounded in the provided chunks only.\n"
        "Return JSON with key 'concepts': array of objects {name, chunk_id}.\n"
        "Each chunk_id MUST be one of the CHUNK_ID values shown in CONTEXT.\n"
        f"Focus on concepts suitable for {difficulty} difficulty questions."
    )
    
    # Build context from retrieval
    ctx_lines: list[str] = []
    take = min(5, max(3, int(getattr(s, "pool_generation_context_chunks", 6) or 6)))
    chunk_chars = max(200, int(getattr(s, "pool_generation_chunk_chars", 900) or 900))
    for r in retrieval_result.chunks[:take]:  # Use top-k for concept extraction
        c = r.chunk
        uri = doc_uri_map.get(c.document_id, "unknown")
        chunk_text = c.text[:chunk_chars]  # Keep small for speed
        ctx_lines.extend([
            f"CHUNK_ID: {c.id}",
            f"CHAPTER: {c.chapter}",
            f"TEXT: {chunk_text[: min(500, len(chunk_text))]}",
            "---",
        ])
    
    user_prompt = "\n".join([
        "CONTEXT:",
        "\n".join(ctx_lines),
        "",
        f"Extract 8-12 examinable concepts suitable for {difficulty} difficulty questions.",
        "For each concept, pick the single best supporting CHUNK_ID from the context.",
        "Return JSON object: {\"concepts\": [{\"name\": \"...\", \"chunk_id\": \"...\"}, ...]}",
    ])
    
    try:
        result = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=_concept_chunk_id_schema(),
        )
        
        parsed = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        concepts = parsed.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []

        allowed_chunk_ids = {str(r.chunk.id) for r in retrieval_result.chunks[:take]}
        out: list[dict] = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            chunk_id = str(c.get("chunk_id") or "").strip()
            if not name:
                continue
            if chunk_id not in allowed_chunk_ids:
                chunk_id = ""
            out.append({"name": name, "chunk_id": chunk_id})

        # Deduplicate by name, keep first
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in out:
            k = item["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(item)

        return deduped[:12] if deduped else [{"name": "General Concepts", "chunk_id": ""}]
        
    except Exception as e:
        log.warning("concept_extraction_failed", error=str(e))
        # Fallback: use chapter names as concepts
        return [{"name": str(ch), "chunk_id": ""} for ch in list({r.chunk.chapter for r in retrieval_result.chunks[:5] if r.chunk.chapter})]


def _generate_questions_for_concept(
    concept: str,
    retrieval,
    doc_uri_map: dict[str, str],
    difficulty: str,
    count: int,
    corpus_id: str,
    preferred_chunk_id: str | None = None,
) -> list[dict]:
    """
    Phase 2: Generate N questions for a specific concept.
    LLM prompt MUST specify difficulty explicitly.
    """
    s = get_settings()
    client = get_llm_client()
    system_prompt = _pool_system_prompt(difficulty)
    
    # Build context with concept focus
    ctx_lines: list[str] = []
    take = max(3, int(getattr(s, "pool_generation_context_chunks", 6) or 6))
    chunk_chars = max(400, int(getattr(s, "pool_generation_chunk_chars", 900) or 900))
    ranked = _rank_chunks_for_concept(retrieval, concept)
    if preferred_chunk_id:
        preferred = [c for c in ranked if str(getattr(c, "id", "")) == str(preferred_chunk_id)]
        rest = [c for c in ranked if str(getattr(c, "id", "")) != str(preferred_chunk_id)]
        ranked = preferred + rest
    picked = ranked[:take] if ranked else [r.chunk for r in (retrieval.chunks or [])[:take]]
    for c in picked:
        uri = doc_uri_map.get(c.document_id, "unknown")
        chunk_text = c.text[:chunk_chars]
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
        f"Generate up to {count} questions about '{concept}'.",
        f"ALL questions must be {difficulty} difficulty.",
        "Question types: mcq, short, long (mix as appropriate).",
        "If the context is insufficient for all questions, generate fewer and add the remainder to 'rejected' with reason=INSUFFICIENT_CONTEXT.",
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
        
        questions = result.parsed_json.get("questions", []) if isinstance(result.parsed_json, dict) else []
        if not isinstance(questions, list):
            log.warning("question_generation_no_questions_array", concept=concept)
            return []
        if not questions:
            log.warning("question_generation_empty", concept=concept)
            return []
        
        # Fix citations
        for q in questions:
            for cit in q.get("citations", []) or []:
                did = cit.get("document_id")
                cit["storage_uri"] = doc_uri_map.get(did, cit.get("storage_uri", ""))

        # Auto-fill missing required fields so we don't end up with 0 questions
        chapter_fallback = (picked[0].chapter if picked and picked[0].chapter else "Unknown")
        citation_chunk = picked[0] if picked else (retrieval.chunks[0].chunk if retrieval.chunks else None)
        if citation_chunk is not None:
            for q in questions:
                if isinstance(q, dict):
                    _ensure_question_minimums(
                        q=q,
                        concept=concept,
                        difficulty=difficulty,
                        chapter_fallback=chapter_fallback,
                        citation_chunk=citation_chunk,
                        doc_uri_map=doc_uri_map,
                    )
        
        # Filter by difficulty and apply guardrails
        valid_questions: list[dict] = []
        rej_counts: dict[str, int] = {}
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
            if rej:
                rej_counts[rej.reason] = rej_counts.get(rej.reason, 0) + 1
            if not rej:
                c0 = (q.get("citations") or [{}])[0]
                rej = reject_if_low_support(
                    answer=q.get("answer"),
                    explanation=q.get("explanation"),
                    excerpt=str(c0.get("excerpt") or ""),
                )
                if rej:
                    rej_counts[rej.reason] = rej_counts.get(rej.reason, 0) + 1
            
            if not rej:
                q["tags"] = {**(q.get("tags") or {}), "chapter": chapter, "dedupe_key": dk, "concept": concept}
                q["tags"]["question_type"] = qtype
                valid_questions.append(q)

        log.info(
            "concept_generation_done",
            concept=concept,
            raw_questions=len(questions),
            valid_questions=len(valid_questions),
            issues_count=len(issues),
            reject_reasons=rej_counts,
        )

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

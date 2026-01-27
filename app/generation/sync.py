from __future__ import annotations

import json

from fastapi import HTTPException
from sqlmodel import select

from app.api.v1.schemas.generation import GenerateRequest, GenerateResponse, GeneratedQuestionOut
from app.db.models import Document
from app.db.session import get_session
from app.generation.guardrails import (
    dedupe_key_for,
    reject_if_embedding_duplicate,
    reject_if_exact_duplicate,
    reject_if_low_support,
)
from app.generation.prompts import build_system_prompt, build_user_prompt
from app.generation.schema import QUESTION_SET_SCHEMA
from app.generation.validation import ensure_mandatory_citations, validate_strict_schema
from app.llm.factory import get_llm_client
from app.observability.logging import log
from app.retrieval.retriever import retrieve
from app.settings import get_settings


def generate_questions_sync(req: GenerateRequest) -> GenerateResponse:
    """
    Generate questions synchronously with retry logic for guardrail rejections.
    Attempts to generate enough questions to meet blueprint requirements.
    """
    s = get_settings()
    bp = req.blueprint
    target_count = bp.count
    top_k = min(req.max_context_chunks or s.max_context_chunks, 50)

    # Mandatory retrieval (single query for sync path).
    query = _blueprint_to_query(bp)
    rr = retrieve(
        corpus_id=req.corpus_id,
        class_level=bp.constraints.class_level,
        subject=bp.constraints.subject,
        chapters=bp.constraints.chapters,
        query=query,
        top_k=top_k,
    )
    if rr.confidence < s.min_retrieval_confidence:
        raise HTTPException(status_code=409, detail="low_retrieval_confidence")

    doc_uri = _document_uri_map(req.corpus_id)
    system_prompt = build_system_prompt()
    client = get_llm_client()

    # Retry logic: attempt up to 2 times (reduced from 3 for faster failure)
    # Most issues are fixed by normalization, so 2 attempts should be enough
    max_attempts = 2
    all_out_questions: list[GeneratedQuestionOut] = []
    all_rejected: list[dict] = []
    
    for attempt in range(max_attempts):
        user_prompt = build_user_prompt(blueprint=bp, retrieval=rr, document_uri_by_id=doc_uri)
        
        try:
            result = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=QUESTION_SET_SCHEMA)
        except Exception as e:
            error_msg = str(e)[:500]  # Limit error message length
            log.error("llm_generation_failed", attempt=attempt + 1, error=error_msg, error_type=type(e).__name__)
            if attempt == max_attempts - 1:
                # Ensure detail is a simple string that can be serialized
                detail_msg = f"generation_failed_after_retries: {error_msg}"
                raise HTTPException(status_code=500, detail=detail_msg)
            continue

        # Normalize common LLM output variants into our strict schema shape
        # This fixes most format issues (options as objects, field name mismatches, etc.)
        _normalize_llm_output(result.parsed_json)

        issues = validate_strict_schema(result.parsed_json) + ensure_mandatory_citations(result.parsed_json)
        if issues:
            # Log the specific issues for debugging
            issue_summary = [f"{i.code}: {i.message}" for i in issues[:3]]  # Show first 3
            log.warning("generation_rejected", attempt=attempt + 1, issues_count=len(issues), issue_summary=issue_summary)
            
            # After normalization, if there are still issues, they're likely unfixable
            # Return partial results if we have some valid questions, otherwise retry once more
            if attempt == max_attempts - 1:
                # If we have some valid questions, return them instead of failing completely
                # This prevents wasting time on retries when normalization can't fix everything
                if all_out_questions:
                    log.warning("generation_partial_after_validation_errors", 
                              valid_questions=len(all_out_questions), 
                              target=target_count,
                              issues_count=len(issues))
                    return GenerateResponse(job_id=None, questions=all_out_questions, rejected=all_rejected)
                raise HTTPException(status_code=422, detail={"code": "INVALID_GENERATION", "issues": [i.__dict__ for i in issues]})
            # Continue to retry (normalization should fix most issues on second attempt)
            continue

        # Fix up citations storage_uri by joining documents
        for q in result.parsed_json.get("questions", []) or []:
            for cit in q.get("citations", []) or []:
                did = cit.get("document_id")
                cit["storage_uri"] = doc_uri.get(did, cit.get("storage_uri"))

        # Guardrails: dedupe + basic support check (reject per-question; keep others)
        attempt_questions: list[GeneratedQuestionOut] = []
        attempt_rejected = list(result.parsed_json.get("rejected", []) or [])
        
        for q in result.parsed_json.get("questions", []) or []:
            stem = str(q.get("stem") or "")
            qtype = str(q.get("qtype") or "")
            chapter = str((q.get("tags") or {}).get("chapter") or bp.constraints.chapters[0])
            dk = dedupe_key_for(stem, qtype, chapter)
            rej = reject_if_exact_duplicate(corpus_id=req.corpus_id, key=dk) or reject_if_embedding_duplicate(
                corpus_id=req.corpus_id, stem=stem
            )
            if not rej:
                # low-support check against first citation excerpt
                c0 = (q.get("citations") or [{}])[0]
                rej = reject_if_low_support(answer=q.get("answer"), explanation=q.get("explanation"), excerpt=str(c0.get("excerpt") or ""))
            if rej:
                attempt_rejected.append({"reason": rej.reason, "details": rej.details})
                continue

            q["tags"] = {**(q.get("tags") or {}), "chapter": chapter, "dedupe_key": dk}
            attempt_questions.append(GeneratedQuestionOut.model_validate(q))
        
        # Add successful questions from this attempt
        all_out_questions.extend(attempt_questions)
        all_rejected.extend(attempt_rejected)
        
        # If we have enough questions, return early
        if len(all_out_questions) >= target_count:
            log.info("generation_success", generated=len(all_out_questions), target=target_count, attempts=attempt + 1)
            return GenerateResponse(
                job_id=None,
                questions=all_out_questions[:target_count],  # Trim to exact count
                rejected=all_rejected,
            )
        
        # If this was the last attempt, return what we have
        if attempt == max_attempts - 1:
            log.warning(
                "generation_partial",
                generated=len(all_out_questions),
                target=target_count,
                attempts=max_attempts,
            )
            break
    
    # Return what we managed to generate (may be less than target)
    return GenerateResponse(job_id=None, questions=all_out_questions, rejected=all_rejected)


def _normalize_llm_output(payload: dict) -> None:
    """
    Best-effort normalization for provider responses that don't exactly match our
    internal schema. This makes the service resilient to small prompt/provider
    format drift while still enforcing strict validation afterward.
    """
    if not isinstance(payload, dict):
        return
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return

    # Define allowed fields per schema to strip extras
    allowed_fields = {
        "id", "qtype", "difficulty", "language", "stem", "options", 
        "answer", "explanation", "tags", "citations"
    }

    for q in questions:
        if not isinstance(q, dict):
            continue

        # Common alternate keys / shapes.
        # Some providers return the Niyukti-style schema:
        # - question_text / question_type / correct_answer
        if "stem" not in q and "question_text" in q:
            q["stem"] = q.pop("question_text")
        
        # Fix common field name mismatches from OpenRouter/OpenAI models
        # "type" -> "qtype"
        if "qtype" not in q and "type" in q:
            q["qtype"] = q.pop("type")
        
        # "citation" (singular) -> "citations" (array)
        if "citations" not in q:
            if "citation" in q:
                # Convert single citation to array
                citation = q.pop("citation")
                q["citations"] = [citation] if isinstance(citation, dict) else []
            else:
                # Ensure citations array exists (will be validated later)
                q["citations"] = []
        if "stem" not in q and "question" in q:
            q["stem"] = q.pop("question")
        if "answer" not in q and "correct_answer" in q:
            q["answer"] = q.pop("correct_answer")
        # Also handle "correct_option" -> "answer" for MCQ
        if "answer" not in q and "correct_option" in q:
            q["answer"] = q.pop("correct_option")

        # Normalize question type keys, including occasional typos.
        # Example typo seen in logs: "queuestuestion_type"
        if "qtype" not in q:
            if "question_type" in q:
                q["qtype"] = q.pop("question_type")
            else:
                # Best-effort: any key that contains "uestion_type"
                for k in list(q.keys()):
                    lk = k.lower()
                    if "uestion_type" in lk:
                        q["qtype"] = q.pop(k)
                        break

        # Normalize qtype values to our enum-like set
        if "qtype" in q and isinstance(q["qtype"], str):
            qt = q["qtype"].strip().lower()
            if qt in {"multiple_choice", "multiple choice", "mcq"}:
                q["qtype"] = "mcq"
            elif qt in {"descriptive", "long_answer", "long"}:
                q["qtype"] = "long"
            elif qt in {"short_answer", "short"}:
                q["qtype"] = "short"

        # Normalize options array: Gemini sometimes returns objects like {'id': 'opt1', 'text': '...'}
        # Schema expects simple strings, so extract 'text' field from option objects
        if "options" in q and isinstance(q["options"], list):
            normalized_options = []
            for opt in q["options"]:
                if isinstance(opt, dict):
                    # Extract 'text' field if present, otherwise use first string value
                    if "text" in opt:
                        normalized_options.append(str(opt["text"]))
                    elif "value" in opt:
                        normalized_options.append(str(opt["value"]))
                    elif "option" in opt:
                        normalized_options.append(str(opt["option"]))
                    else:
                        # Try to find any string value in the object
                        for val in opt.values():
                            if isinstance(val, str):
                                normalized_options.append(val)
                                break
                elif isinstance(opt, str):
                    normalized_options.append(opt)
                else:
                    # Convert to string as last resort
                    normalized_options.append(str(opt))
            q["options"] = normalized_options

        # Strip extra fields that aren't in schema (e.g., "chapter", "correct_option")
        # This prevents schema validation failures from Gemini adding extra fields
        extra_fields = [k for k in q.keys() if k not in allowed_fields]
        for field in extra_fields:
            # If it's "chapter", try to move it to tags.chapter if tags exists
            if field == "chapter" and isinstance(q.get("tags"), dict):
                q.setdefault("tags", {})["chapter"] = q.pop(field)
            else:
                # Remove other extra fields
                q.pop(field, None)

        # Ensure required keys exist (even if empty) so schema validation can
        # report meaningful issues later.
        q.setdefault("language", "en")
        q.setdefault("tags", {})
        q.setdefault("difficulty", "easy")


def _blueprint_to_query(bp) -> str:
    js = bp.job_summary or {}
    skills = " ".join((js.get("skills") or []))
    role = str(js.get("role") or "")
    focus = " ".join((js.get("focus_topics") or []))
    return f"CBSE {bp.constraints.subject} class {bp.constraints.class_level}. Chapters: {', '.join(bp.constraints.chapters)}. {role}. {skills}. {focus}"


def _document_uri_map(corpus_id: str) -> dict[str, str]:
    with get_session() as session:
        docs = session.exec(select(Document).where(Document.corpus_id == corpus_id)).all()
    return {d.id: d.storage_uri for d in docs}



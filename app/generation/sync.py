from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select

from app.api.v1.schemas.generation import GenerateRequest, GenerateResponse, GeneratedQuestionOut
from app.db.models import Chunk, Document
from app.db.session import get_session_ctx
from app.generation.guardrails import (
    GuardrailReject,
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
from app.retrieval.retriever import _normalize_subject, retrieve
from app.settings import get_settings


GENERATION_BATCH_SIZE = None
GENERATION_BATCH_CONCURRENCY = 4
JOB_TEXT_SIMILARITY_THRESHOLD = 0.96


def _summarize_rejections(rejected: list[dict], *, limit: int = 8) -> dict[str, int]:
    if not rejected:
        return {}
    counts = Counter(str(item.get("reason") or "UNKNOWN") for item in rejected if isinstance(item, dict))
    return dict(counts.most_common(limit))


async def generate_questions_sync(req: GenerateRequest, *, document_id: str | None = None) -> GenerateResponse:
    """
    Generate questions synchronously with retry logic for guardrail rejections.
    Attempts to generate enough questions to meet blueprint requirements.
    """
    s = get_settings()
    batch_size = s.generation_batch_size
    relaxed_validation = bool(getattr(s, "generation_debug_relaxed_validation", False))
    disable_guardrails = bool(getattr(s, "generation_disable_guardrails", False)) or relaxed_validation
    max_total_attempts = max(1, int(getattr(s, "generation_max_total_attempts", 48)))
    max_batches = max(1, int(getattr(s, "generation_max_batches", 64)))
    attempt_budget = {"remaining": max_total_attempts}
    bp = req.blueprint
    target_count = bp.count
    top_k = min(
        bp.count * s.generation_max_chunks_per_question,
        s.generation_max_retrieval_cap,
    )

    doc_uri = await _document_uri_map(req.corpus_id)
    system_prompt = build_system_prompt()
    client = get_llm_client()
    job_dedupe_state = _JobDedupeState()

    def _normalize_stem_text(text: str) -> str:
        norm = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def _reserve_job_dedupe(
        *,
        norm_stem: str,
        key: str,
        similarity_threshold: float,
    ) -> GuardrailReject | None:
        if not norm_stem:
            return None
        if norm_stem in job_dedupe_state.seen_norm or norm_stem in job_dedupe_state.inflight_norm:
            return GuardrailReject(
                reason="DUPLICATE_QUESTION",
                details={"match_type": "job_text_exact"},
            )
        active = list(job_dedupe_state.seen_norm) + list(job_dedupe_state.inflight_norm)
        for other in active:
            if other and SequenceMatcher(None, norm_stem, other).ratio() >= similarity_threshold:
                return GuardrailReject(
                    reason="DUPLICATE_QUESTION",
                    details={"match_type": "job_text_similar", "threshold": similarity_threshold},
                )
        if key in job_dedupe_state.seen_keys or key in job_dedupe_state.inflight_keys:
            return GuardrailReject(
                reason="DUPLICATE_QUESTION",
                details={"match_type": "job_key"},
            )
        job_dedupe_state.inflight_norm.add(norm_stem)
        job_dedupe_state.inflight_keys.add(key)
        return None

    def _finalize_job_dedupe(*, norm_stem: str, key: str, accepted: bool) -> None:
        if not norm_stem:
            return
        job_dedupe_state.inflight_norm.discard(norm_stem)
        job_dedupe_state.inflight_keys.discard(key)
        if accepted:
            job_dedupe_state.seen_norm.add(norm_stem)
            job_dedupe_state.seen_keys.add(key)

    def _build_batches(mix, batch_size: int) -> list[dict[str, int]]:
        remaining = {
            "easy": int(mix.easy),
            "medium": int(mix.medium),
            "hard": int(mix.hard),
        }
        seq: list[str] = []
        while sum(remaining.values()) > 0:
            for diff in ("easy", "medium", "hard"):
                if remaining[diff] > 0:
                    seq.append(diff)
                    remaining[diff] -= 1
        batches: list[dict[str, int]] = []
        for i in range(0, len(seq), batch_size):
            batch = {"easy": 0, "medium": 0, "hard": 0}
            for diff in seq[i : i + batch_size]:
                batch[diff] += 1
            batches.append(batch)
        return batches

    _retrieval_cache: dict[str, object] = {}

    async def _retrieve_cached(bp_local: GenerateRequest):
        """Retrieve chunks once per chapter key and cache the result."""
        chapters_key = "|".join(sorted(bp_local.blueprint.constraints.chapters))
        cache_key = f"{bp_local.corpus_id}:{bp_local.blueprint.constraints.class_level}:{bp_local.blueprint.constraints.subject}:{chapters_key}"
        if cache_key in _retrieval_cache:
            return _retrieval_cache[cache_key]

        query = _blueprint_to_query(bp_local.blueprint)
        chapter_number = bp_local.blueprint.constraints.chapter_number
        if chapter_number is None and len(bp_local.blueprint.constraints.chapters) == 1:
            ch_only = str(bp_local.blueprint.constraints.chapters[0]).strip()
            if ch_only.isdigit():
                chapter_number = int(ch_only)
        rr = await retrieve(
            corpus_id=bp_local.corpus_id,
            document_id=document_id,
            class_level=bp_local.blueprint.constraints.class_level,
            subject=bp_local.blueprint.constraints.subject,
            chapters=bp_local.blueprint.constraints.chapters,
            query=query,
            top_k=max(top_k, 30),
            board=bp_local.blueprint.constraints.board,
            syllabus_version=bp_local.blueprint.constraints.syllabus_version,
            chapter_number=chapter_number,
        )
        _retrieval_cache[cache_key] = rr
        return rr

    async def _generate_for_blueprint(
        bp_local: GenerateRequest,
    ) -> tuple[list[GeneratedQuestionOut], list[dict]]:
        rr = await _retrieve_cached(bp_local)
        if not rr.chunks:
            return [], [{"reason": "INSUFFICIENT_CONTEXT", "details": {"code": "no_in_scope_chunks"}}]
        if rr.confidence < s.min_retrieval_confidence:
            return [], [
                {
                    "reason": "LOW_RETRIEVAL_CONFIDENCE",
                    "details": {"confidence": rr.confidence, "threshold": s.min_retrieval_confidence},
                }
            ]

        out_questions: list[GeneratedQuestionOut] = []
        rejected: list[dict] = []

        if attempt_budget["remaining"] <= 0:
            rejected.append(
                {
                    "reason": "ATTEMPT_BUDGET_EXHAUSTED",
                    "details": {
                        "max_total_attempts": max_total_attempts,
                        "scope": "generate_questions_sync",
                    },
                }
            )
            log.warning(
                "generation_attempt_budget_exhausted",
                max_total_attempts=max_total_attempts,
                generated=0,
                rejected=len(rejected),
            )
            return out_questions, rejected
        attempt_budget["remaining"] -= 1
        attempt_num_global = max_total_attempts - attempt_budget["remaining"]

        user_prompt = build_user_prompt(
            blueprint=bp_local.blueprint, retrieval=rr, document_uri_by_id=doc_uri
        )

        try:
            if hasattr(client, "async_generate_json"):
                result = await client.async_generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_schema=QUESTION_SET_SCHEMA,
                )
            else:
                result = await asyncio.to_thread(
                    client.generate_json,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_schema=QUESTION_SET_SCHEMA,
                )
        except Exception as e:
            error_msg = str(e)[:500]
            log.error(
                "llm_generation_failed",
                attempt=1,
                error=error_msg,
                error_type=type(e).__name__,
            )
            raise HTTPException(status_code=500, detail=f"generation_failed: {error_msg}")

        _normalize_llm_output(result.parsed_json)

        issues = validate_strict_schema(result.parsed_json) + ensure_mandatory_citations(result.parsed_json)
        if issues and not relaxed_validation:
            issue_summary = [f"{i.code}: {i.message}" for i in issues[:3]]
            rejected.extend(
                [
                    {
                        "reason": "INVALID_SCHEMA",
                        "details": {
                            "code": i.code,
                            "message": i.message,
                            "path": (i.details or {}).get("path"),
                        },
                    }
                    for i in issues
                ]
            )
            log.warning(
                "generation_rejected",
                attempt=1,
                global_attempt=attempt_num_global,
                issues_count=len(issues),
                issue_summary=issue_summary,
            )
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_GENERATION", "issues": [i.__dict__ for i in issues]},
            )
        if issues and relaxed_validation:
            issue_summary = [f"{i.code}: {i.message}" for i in issues[:3]]
            log.warning(
                "generation_validation_relaxed",
                attempt=1,
                global_attempt=attempt_num_global,
                issues_count=len(issues),
                issue_summary=issue_summary,
            )

        for q in result.parsed_json.get("questions", []) or []:
            for cit in q.get("citations", []) or []:
                did = cit.get("document_id")
                cit["storage_uri"] = doc_uri.get(did, cit.get("storage_uri"))

        attempt_rejected = list(result.parsed_json.get("rejected", []) or [])
        fallback_citation = None
        if rr.chunks:
            first = rr.chunks[0].chunk
            first_document_id = str(first.document_id)
            fallback_citation = {
                "chunk_id": str(first.id),
                "document_id": first_document_id,
                "storage_uri": str(doc_uri.get(first_document_id, "")),
                "page_start": int(first.page_start),
                "page_end": int(first.page_end),
                "excerpt": str(first.text or "")[:400] or "Context excerpt unavailable.",
            }

        for q in result.parsed_json.get("questions", []) or []:
            q.setdefault("id", str(uuid4()))
            q.setdefault("qtype", str(bp_local.blueprint.question_types[0] if bp_local.blueprint.question_types else "mcq"))
            q.setdefault("difficulty", "medium")
            q.setdefault("language", bp_local.blueprint.constraints.language or "en")
            q.setdefault("tags", {})
            q.setdefault("stem", str(q.get("question") or "").strip() or "Generated question.")
            if relaxed_validation and not q.get("citations") and fallback_citation:
                q["citations"] = [fallback_citation]

            stem = str(q.get("stem") or "")
            qtype = str(q.get("qtype") or "")
            chapter = str(
                (q.get("tags") or {}).get("chapter")
                or bp_local.blueprint.constraints.chapters[0]
            )
            dk = dedupe_key_for(stem, qtype, chapter)
            norm_stem = _normalize_stem_text(stem)
            rej = None
            if not disable_guardrails:
                rej = _reserve_job_dedupe(
                    norm_stem=norm_stem,
                    key=dk,
                    similarity_threshold=JOB_TEXT_SIMILARITY_THRESHOLD,
                )
            if rej:
                attempt_rejected.append({"reason": rej.reason, "details": rej.details})
                continue

            c0 = (q.get("citations") or [{}])[0]
            if not disable_guardrails:
                rej = await reject_if_exact_duplicate(corpus_id=req.corpus_id, key=dk)
                if not rej:
                    rej = reject_if_low_support(
                        answer=q.get("answer"),
                        explanation=q.get("explanation"),
                        excerpt=str(c0.get("excerpt") or ""),
                    )
                if not rej:
                    rej = await reject_if_embedding_duplicate(corpus_id=req.corpus_id, stem=stem)
            if rej:
                _finalize_job_dedupe(norm_stem=norm_stem, key=dk, accepted=False)
                attempt_rejected.append({"reason": rej.reason, "details": rej.details})
                continue

            q["tags"] = {
                **(q.get("tags") or {}),
                "chapter": chapter,
                "question_type": qtype,
                "dedupe_key": dk,
            }
            try:
                out_questions.append(GeneratedQuestionOut.model_validate(q))
                if not disable_guardrails:
                    _finalize_job_dedupe(norm_stem=norm_stem, key=dk, accepted=True)
            except Exception as exc:  # noqa: BLE001
                attempt_rejected.append(
                    {
                        "reason": "MODEL_VALIDATION_FAILED",
                        "details": {"error": str(exc)[:200]},
                    }
                )
                if not disable_guardrails:
                    _finalize_job_dedupe(norm_stem=norm_stem, key=dk, accepted=False)
                continue

        rejected.extend(attempt_rejected)
        log.info(
            "generation_attempt_summary",
            attempt=1,
            global_attempt=attempt_num_global,
            accepted=len(out_questions),
            rejected=len(attempt_rejected),
            reject_reason_counts=_summarize_rejections(attempt_rejected),
            relaxed_validation=relaxed_validation,
            guardrails_enabled=not disable_guardrails,
        )

        return out_questions[: bp_local.blueprint.count], rejected

    async def _run_batches_for_request(
        local_req: GenerateRequest,
    ) -> tuple[list[GeneratedQuestionOut], list[dict], Exception | None]:
        local_bp = local_req.blueprint
        local_target = local_bp.count
        batches = _build_batches(
            local_bp.difficulty_mix, min(batch_size, local_target)
        )
        original_batch_count = len(batches)
        if original_batch_count > max_batches:
            log.warning(
                "generation_batch_budget_trimmed",
                original_batches=original_batch_count,
                max_batches=max_batches,
                target=local_target,
            )
            batches = batches[:max_batches]
        out: list[GeneratedQuestionOut] = []
        rejected: list[dict] = []
        if original_batch_count > max_batches:
            rejected.append(
                {
                    "reason": "BATCH_BUDGET_EXHAUSTED",
                    "details": {"max_batches": max_batches, "target": local_target},
                }
            )
        last_error: Exception | None = None

        max_concurrency = min(GENERATION_BATCH_CONCURRENCY, len(batches))
        if max_concurrency <= 1:
            for idx, batch_mix in enumerate(batches, start=1):
                if len(out) >= local_target:
                    break
                batch_count = sum(batch_mix.values())
                batch_bp = local_bp.model_copy(deep=True)
                batch_bp.count = batch_count
                batch_bp.difficulty_mix = type(local_bp.difficulty_mix)(**batch_mix)
                batch_req = local_req.model_copy(deep=True)
                batch_req.blueprint = batch_bp

                log.info(
                    "generation_batch_start",
                    batch=idx,
                    batch_count=batch_count,
                    remaining=local_target - len(out),
                )

                try:
                    batch_questions, batch_rejected = await _generate_for_blueprint(batch_req)
                    out.extend(batch_questions)
                    rejected.extend(batch_rejected)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    log.warning(
                        "generation_batch_failed",
                        batch=idx,
                        error=str(exc)[:500],
                        error_type=type(exc).__name__,
                    )
                    continue
            return out[:local_target], rejected, last_error

        tasks = []
        for idx, mix in enumerate(batches, start=1):
            batch_count = sum(mix.values())
            batch_bp = local_bp.model_copy(deep=True)
            batch_bp.count = batch_count
            batch_bp.difficulty_mix = type(local_bp.difficulty_mix)(**mix)
            batch_req = local_req.model_copy(deep=True)
            batch_req.blueprint = batch_bp

            log.info(
                "generation_batch_start",
                batch=idx,
                batch_count=batch_count,
                remaining=local_target - len(out),
            )

            tasks.append(_generate_for_blueprint(batch_req))

        semaphore = asyncio.Semaphore(GENERATION_BATCH_CONCURRENCY)

        async def limited_task(coro):
            async with semaphore:
                return await coro

        limited_tasks = [limited_task(t) for t in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        for idx, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                last_error = result
                log.warning(
                    "generation_batch_failed",
                    batch=idx,
                    error=str(result)[:500],
                    error_type=type(result).__name__,
                )
                continue
            batch_questions, batch_rejected = result
            out.extend(batch_questions)
            rejected.extend(batch_rejected)

        return out[:local_target], rejected, last_error

    async def _expand_chapters_for_corpus(
        chapters: list[str],
        *,
        corpus_id: str,
        document_id: str | None,
        class_level: int,
        subject: str,
    ) -> list[str]:
        async with get_session_ctx() as session:
            q = (
                select(Chunk.chapter)
                .where(Chunk.corpus_id == corpus_id)
                .where(Chunk.class_level == class_level)
                .where(Chunk.subject == _normalize_subject(subject))
            )
            if document_id:
                q = q.where(Chunk.document_id == document_id)
            rows = (await session.execute(q.distinct())).scalars().all()
        raw = [str(r).strip() for r in rows if r and str(r).strip()]
        if not raw:
            return chapters

        def _chapter_sort_key(ch: str) -> tuple[int, str]:
            m = re.search(r"(?:chapter|unit|ch)\s*(\d+)", ch.lower())
            if m:
                return (int(m.group(1)), ch)
            if ch.isdigit():
                return (int(ch), ch)
            return (10**9, ch)

        return sorted(list(dict.fromkeys(raw)), key=_chapter_sort_key)

    def _split_difficulty_mix(mix, chapters: list[str]) -> dict[str, dict[str, int]]:
        per_chapter = {ch: {"easy": 0, "medium": 0, "hard": 0} for ch in chapters}
        if not chapters:
            return per_chapter
        for diff in ("easy", "medium", "hard"):
            total = int(getattr(mix, diff))
            base = total // len(chapters)
            remainder = total % len(chapters)
            for idx, ch in enumerate(chapters):
                per_chapter[ch][diff] = base + (1 if idx < remainder else 0)
        return per_chapter

    if bp.constraints.per_chapter and not bp.constraints.chapters:
        expanded = await _expand_chapters_for_corpus(
            bp.constraints.chapters,
            corpus_id=req.corpus_id,
            document_id=document_id,
            class_level=bp.constraints.class_level,
            subject=bp.constraints.subject,
        )
        if expanded:
            bp.constraints.chapters = expanded

    if bp.constraints.per_chapter and len(bp.constraints.chapters) > 1:
        all_out_questions: list[GeneratedQuestionOut] = []
        all_rejected: list[dict] = []
        last_error: Exception | None = None
        per_chapter_mix = _split_difficulty_mix(bp.difficulty_mix, bp.constraints.chapters)

        for chapter in bp.constraints.chapters:
            chapter_mix = per_chapter_mix.get(chapter, {"easy": 0, "medium": 0, "hard": 0})
            chapter_count = sum(chapter_mix.values())
            if chapter_count <= 0:
                continue
            chapter_bp = bp.model_copy(deep=True)
            chapter_bp.count = chapter_count
            chapter_bp.difficulty_mix = type(bp.difficulty_mix)(**chapter_mix)
            chapter_bp.constraints.chapters = [chapter]
            chapter_req = req.model_copy(deep=True)
            chapter_req.blueprint = chapter_bp

            chapter_questions, chapter_rejected, chapter_error = await _run_batches_for_request(chapter_req)
            all_out_questions.extend(chapter_questions)
            all_rejected.extend(chapter_rejected)
            if chapter_error:
                last_error = chapter_error

        if len(all_out_questions) >= target_count:
            log.info(
                "generation_success",
                generated=len(all_out_questions),
                target=target_count,
                attempts=len(bp.constraints.chapters),
            )
            return GenerateResponse(
                job_id=None,
                questions=all_out_questions[:target_count],
                rejected=all_rejected,
            )

        log.warning(
            "generation_partial",
            generated=len(all_out_questions),
            target=target_count,
            attempts=len(bp.constraints.chapters),
            reject_reason_counts=_summarize_rejections(all_rejected),
        )
        if not all_out_questions and last_error:
            raise last_error
        return GenerateResponse(job_id=None, questions=all_out_questions, rejected=all_rejected)

    all_out_questions, all_rejected, last_error = await _run_batches_for_request(req)
    if len(all_out_questions) >= target_count:
        log.info(
            "generation_success",
            generated=len(all_out_questions),
            target=target_count,
            attempts=1,
        )
        return GenerateResponse(
            job_id=None,
            questions=all_out_questions[:target_count],
            rejected=all_rejected,
        )

    log.warning(
        "generation_partial",
        generated=len(all_out_questions),
        target=target_count,
        attempts=1,
        reject_reason_counts=_summarize_rejections(all_rejected),
    )
    if not all_out_questions and last_error:
        raise last_error
    return GenerateResponse(job_id=None, questions=all_out_questions, rejected=all_rejected)


class _JobDedupeState:
    def __init__(self) -> None:
        self.seen_norm: set[str] = set()
        self.seen_keys: set[str] = set()
        self.inflight_norm: set[str] = set()
        self.inflight_keys: set[str] = set()


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

    allowed_fields = {
        "id", "qtype", "difficulty", "language", "stem", "options", 
        "answer", "explanation", "tags", "citations"
    }

    for q in questions:
        if not isinstance(q, dict):
            continue

        if "stem" not in q and "question_text" in q:
            q["stem"] = q.pop("question_text")
        
        if "qtype" not in q and "type" in q:
            q["qtype"] = q.pop("type")
        
        if "citations" not in q:
            if "citation" in q:
                citation = q.pop("citation")
                q["citations"] = [citation] if isinstance(citation, dict) else []
            else:
                q["citations"] = []
        if "stem" not in q and "question" in q:
            q["stem"] = q.pop("question")
        if "answer" not in q and "correct_answer" in q:
            q["answer"] = q.pop("correct_answer")
        if "answer" not in q and "correct_option" in q:
            q["answer"] = q.pop("correct_option")

        if "qtype" not in q:
            if "question_type" in q:
                q["qtype"] = q.pop("question_type")
            else:
                for k in list(q.keys()):
                    lk = k.lower()
                    if "uestion_type" in lk:
                        q["qtype"] = q.pop(k)
                        break

        if "qtype" in q and isinstance(q["qtype"], str):
            qt = q["qtype"].strip().lower()
            if qt in {"multiple_choice", "multiple choice", "mcq"}:
                q["qtype"] = "mcq"
            elif qt in {"descriptive", "long_answer", "long"}:
                q["qtype"] = "long"
            elif qt in {"short_answer", "short"}:
                q["qtype"] = "short"

        if "options" in q and isinstance(q["options"], list):
            normalized_options = []
            for opt in q["options"]:
                if isinstance(opt, dict):
                    if "text" in opt:
                        normalized_options.append(str(opt["text"]))
                    elif "value" in opt:
                        normalized_options.append(str(opt["value"]))
                    elif "option" in opt:
                        normalized_options.append(str(opt["option"]))
                    else:
                        for val in opt.values():
                            if isinstance(val, str):
                                normalized_options.append(val)
                                break
                elif isinstance(opt, str):
                    normalized_options.append(opt)
                else:
                    normalized_options.append(str(opt))
            q["options"] = normalized_options

        extra_fields = [k for k in q.keys() if k not in allowed_fields]
        for field in extra_fields:
            if field == "chapter" and isinstance(q.get("tags"), dict):
                q.setdefault("tags", {})["chapter"] = q.pop(field)
            else:
                q.pop(field, None)

        q.setdefault("language", "en")
        q.setdefault("tags", {})
        q.setdefault("difficulty", "easy")


def _blueprint_to_query(bp) -> str:
    js = bp.job_summary or {}
    skills = " ".join((js.get("skills") or []))
    role = str(js.get("role") or "")
    focus = " ".join((js.get("focus_topics") or []))
    return f"CBSE {bp.constraints.subject} class {bp.constraints.class_level}. Chapters: {', '.join(bp.constraints.chapters)}. {role}. {skills}. {focus}"


async def _document_uri_map(corpus_id: str) -> dict[str, str]:
    async with get_session_ctx() as session:
        docs = (await session.execute(select(Document).where(Document.corpus_id == corpus_id))).scalars().all()
    return {d.id: d.storage_uri for d in docs}

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.guards import assert_corpus_owner
from app.api.v1.schemas.school import (
    GenerationMeta,
    SchoolGenerateRequest,
    SchoolGenerateResponse,
    SchoolQuestion,
)
from app.db.session import get_session_ctx
from app.generation.prompts import build_school_question_prompt
from app.generation.validation import parse_questions_from_llm_response
from app.llm.factory import get_llm_client
from app.observability.logging import log
from app.vector_store.search import search_chunks

router = APIRouter(tags=["school"])

SCHOOL_QUESTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "type": {"type": "string", "enum": ["mcq", "long"]},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_chapter": {"type": "string"},
                },
                "required": ["question", "type", "difficulty", "correct_answer"],
            },
        }
    },
    "required": ["questions"],
}


@router.post("/generate-questions", response_model=SchoolGenerateResponse)
async def school_generate_questions(
    body: SchoolGenerateRequest,
    request: Request,
):
    tenant_id: str = request.state.tenant_id

    async with get_session_ctx() as session:
        corpus = await assert_corpus_owner(body.corpus_id, tenant_id, session)

    start_ms = int(time.time() * 1000)

    all_questions: list[SchoolQuestion] = []

    difficulty_buckets = [
        ("easy", body.difficulty_mix.easy),
        ("medium", body.difficulty_mix.medium),
        ("hard", body.difficulty_mix.hard),
    ]

    for difficulty, count in difficulty_buckets:
        if count == 0:
            continue

        chunks = await asyncio.to_thread(
            search_chunks,
            query=f"{body.subject or ''} {body.chapter or ''} {difficulty} questions".strip(),
            tenant_id=tenant_id,
            corpus_id=body.corpus_id,
            top_k=min(count * 3, 20),
            strict_metadata_filter=False,
        )

        if not chunks:
            raise HTTPException(
                422,
                f"No indexed content found for difficulty={difficulty}. "
                "Ensure PDF is fully indexed.",
            )

        prompt = build_school_question_prompt(
            chunks=chunks,
            count=count,
            difficulty=difficulty,
            question_type=body.question_type,
            subject=body.subject,
            chapter=body.chapter,
            language=body.language,
        )

        llm = get_llm_client()
        try:
            result = await asyncio.to_thread(
                llm.generate_json,
                system_prompt="You are an expert teacher creating exam questions. Return valid JSON only.",
                user_prompt=prompt,
                json_schema=SCHOOL_QUESTION_JSON_SCHEMA,
            )
            raw_response = result.raw_text
        except RuntimeError as exc:
            log.error("school_llm_error", difficulty=difficulty, error=str(exc)[:500])
            raise HTTPException(502, f"LLM generation failed for difficulty={difficulty}") from exc

        parsed = parse_questions_from_llm_response(
            raw=raw_response,
            difficulty=difficulty,
            question_type=body.question_type,
            language=body.language,
        )
        all_questions.extend(parsed)

    end_ms = int(time.time() * 1000)

    return SchoolGenerateResponse(
        corpus_id=body.corpus_id,
        tenant_id=tenant_id,
        questions=all_questions,
        meta=GenerationMeta(
            easy_count=len([q for q in all_questions if q.difficulty == "easy"]),
            medium_count=len([q for q in all_questions if q.difficulty == "medium"]),
            hard_count=len([q for q in all_questions if q.difficulty == "hard"]),
            total_generated=len(all_questions),
            generation_time_ms=end_ms - start_ms,
        ),
    )

"""Runtime RAG API: POST /rag/generate/questions."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas.rag import RAGGenerateRequest, RAGGenerateResponse, RAGQuestionOut
from app.observability.logging import log
from rag.query.generate_questions import generate_questions

router = APIRouter()


@router.post("/generate/questions", response_model=RAGGenerateResponse)
async def rag_generate_questions(req: RAGGenerateRequest) -> RAGGenerateResponse:
    """
    Generate assessment questions from Qdrant-backed RAG.

    Validates required fields, retrieves chunks with strict filters,
    calls LLM with grounded prompt, returns structured questions only.
    """
    try:
        result = await asyncio.to_thread(
            generate_questions,
            board=req.board,
            class_val=req.class_,
            subject=req.subject,
            chapter_number=req.chapter_number,
            difficulty=req.difficulty,
            question_type=req.question_type,
            count=req.count,
        )
    except ValueError as e:
        log.warning("rag_validate_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        log.error("rag_runtime_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        log.exception("rag_unexpected_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=502,
            detail=f"RAG pipeline error: {type(e).__name__}: {str(e)[:500]}",
        ) from e

    questions = [
        RAGQuestionOut(
            stem=q["stem"],
            options=q.get("options"),
            answer=q.get("answer"),
            qtype=q["qtype"],
            difficulty=q["difficulty"],
        )
        for q in result
    ]
    return RAGGenerateResponse(questions=questions)

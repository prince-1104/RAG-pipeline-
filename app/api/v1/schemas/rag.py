"""Request/response schemas for runtime RAG generate-questions API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["mcq", "descriptive"]


class RAGGenerateRequest(BaseModel):
    """POST /rag/generate/questions body."""

    board: str = Field(..., min_length=1, description="Board identifier (e.g. CBSE)")
    class_: int = Field(..., ge=1, le=12, alias="class", description="Class level 1–12")
    subject: str = Field(..., min_length=1, description="Subject name")
    chapter_number: int = Field(..., ge=1, description="Chapter number")
    difficulty: Difficulty = Field(..., description="easy | medium | hard")
    question_type: QuestionType = Field(..., description="mcq | descriptive")
    count: int = Field(..., ge=1, le=50, description="Number of questions to generate")

    model_config = {"populate_by_name": True}


class RAGQuestionOut(BaseModel):
    """Single generated question."""

    stem: str
    options: list[str] | None = None
    answer: str | None = None
    qtype: Literal["mcq", "descriptive"]
    difficulty: Literal["easy", "medium", "hard"]


class RAGGenerateResponse(BaseModel):
    """POST /rag/generate/questions response."""

    questions: list[RAGQuestionOut] = Field(default_factory=list)

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.api.v1.schemas.common import Citation


Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["mcq", "short", "long", "case_based", "assertion_reason"]


class DifficultyMix(BaseModel):
    easy: int = 0
    medium: int = 0
    hard: int = 0


class QuestionConstraints(BaseModel):
    board: str | None = "CBSE"
    class_level: int = Field(..., ge=1, le=12)
    subject: str
    chapters: list[str] = Field(..., min_length=1)
    chapter_number: int | None = None
    syllabus_version: str | None = None
    language: Literal["en", "hi"] = "en"
    per_chapter: bool = False


class Blueprint(BaseModel):
    blueprint_version: str = "2026-01-02"

    # integration boundary: Niyukti provides a "job requirement" summary, not raw DB ids
    job_summary: dict[str, Any] = Field(
        ...,
        description="Normalized job requirements from Niyukti (skills/roles/topics); opaque to this service except for text fields.",
    )

    constraints: QuestionConstraints
    question_types: list[QuestionType] = Field(..., min_length=1)
    count: int = Field(..., ge=1, le=200)
    difficulty_mix: DifficultyMix
    coverage_strategy: Literal["balanced", "chapter_weighted", "focus_weak_areas"] = "balanced"


class GenerateRequest(BaseModel):
    tenant_id: str
    corpus_id: str
    blueprint: Blueprint

    # cost controls
    max_context_chunks: int | None = Field(default=None, ge=1, le=50)


class GeneratedQuestionOut(BaseModel):
    id: str
    qtype: QuestionType
    difficulty: Difficulty
    language: str

    stem: str
    options: list[str] | None = None
    answer: str | None = None
    explanation: str | None = None

    tags: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation]


class GenerateResponse(BaseModel):
    job_id: str | None = None
    questions: list[GeneratedQuestionOut] | None = None
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class GenerateJobResponse(BaseModel):
    job_id: str
    status: str



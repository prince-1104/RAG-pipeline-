from __future__ import annotations

from pydantic import BaseModel, Field


class NiyuktiGenerateRequest(BaseModel):
    """Simplified request format expected by Niyukti platform"""
    board: str | None = Field(
        None, description="Board identifier (e.g., 'CBSE'). Required if X-Corpus-ID is missing."
    )
    class_level: int = Field(..., ge=1, le=12, description="Class level (1-12)")
    subject: str = Field(..., description="Subject name (e.g., 'maths', 'science')")
    chapter: str | int | None = Field(
        None, description="Chapter identifier (string or int). Required if X-Corpus-ID is missing."
    )
    question_count: int | None = Field(
        None, ge=0, le=200, description="Total number of questions (used if counts not provided)."
    )
    mcq_count: int = Field(0, ge=0, le=200, description="Number of MCQ questions (used if difficulty counts not provided)")
    descriptive_count: int = Field(0, ge=0, le=200, description="Number of descriptive questions (used if difficulty counts not provided)")
    difficulty_level: str | None = Field(None, description="Optional: 'easy', 'medium', or 'hard' (used if difficulty counts not provided)")
    # Per-difficulty counts (from Niyukti UI)
    easy_count: int | None = Field(None, ge=0, le=200, description="Number of easy questions (if provided, overrides auto-distribution)")
    medium_count: int | None = Field(None, ge=0, le=200, description="Number of medium questions (if provided, overrides auto-distribution)")
    hard_count: int | None = Field(None, ge=0, le=200, description="Number of hard questions (if provided, overrides auto-distribution)")
    mcq_ratio: float | None = Field(None, ge=0, le=100, description="MCQ percentage (0-100). If provided, applies to each difficulty level separately")


class NiyuktiQuestionResponse(BaseModel):
    """Question format expected by Niyukti platform"""
    question_text: str
    question_type: str  # "mcq" or "descriptive"
    options: list[str] | None = None  # For MCQ only
    correct_answer: str | None = None  # For MCQ: option text, for descriptive: expected answer
    expected_keywords: list[str] | None = None  # For descriptive only
    expected_answer_text: str | None = None  # For descriptive only
    class_level: int
    subject: str


class NiyuktiGenerateResponse(BaseModel):
    """Response format expected by Niyukti platform"""
    questions: list[NiyuktiQuestionResponse]

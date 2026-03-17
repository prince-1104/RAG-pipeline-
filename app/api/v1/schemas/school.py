from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DifficultyMix(BaseModel):
    easy: int = Field(ge=0)
    medium: int = Field(ge=0)
    hard: int = Field(ge=0)


class SchoolGenerateRequest(BaseModel):
    corpus_id: str
    difficulty_mix: DifficultyMix
    subject: Optional[str] = None
    chapter: Optional[str] = None
    question_type: Literal["mcq", "long", "mixed"] = "mcq"
    language: str = "en"

    @property
    def total_questions(self) -> int:
        return self.difficulty_mix.easy + self.difficulty_mix.medium + self.difficulty_mix.hard


class SchoolQuestion(BaseModel):
    id: str
    question: str
    type: Literal["mcq", "long"]
    difficulty: Literal["easy", "medium", "hard"]
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: Optional[str] = None
    source_chapter: Optional[str] = None
    language: str = "en"


class GenerationMeta(BaseModel):
    easy_count: int
    medium_count: int
    hard_count: int
    total_generated: int
    generation_time_ms: int


class SchoolGenerateResponse(BaseModel):
    corpus_id: str
    tenant_id: str
    questions: List[SchoolQuestion]
    meta: GenerationMeta

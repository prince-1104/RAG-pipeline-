from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PoolRequest(BaseModel):
    """Request to fetch questions from a pool."""
    tenant_id: str
    corpus_id: str
    syllabus: str = "CBSE"
    class_level: int = Field(..., ge=1, le=12)
    subject: str
    difficulty_mix: dict[Literal["easy", "medium", "hard"], int] = Field(
        ..., description="Dictionary mapping difficulty to count, e.g. {'easy': 5, 'medium': 3, 'hard': 2}"
    )
    syllabus_version: str = "2024"


class PoolResponse(BaseModel):
    """Response from pool-based question serving."""
    questions: list[dict] | None = None
    status: Literal["ready", "generating", "pending"]
    pool_info: dict[str, str] | None = None  # Pool IDs per difficulty

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AdminGenerateQuestionsRequest(BaseModel):
    """
    Admin-triggered question generation request.
    Admin can specify criteria for generating questions from uploaded PDFs.
    """
    document_id: str = Field(..., description="Document ID for which to generate questions")
    from_class: int = Field(..., ge=1, le=12, description="Starting class level")
    to_class: int = Field(..., ge=1, le=12, description="Ending class level (inclusive)")
    subject: str = Field(..., description="Subject name")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level")
    question_count: int = Field(..., ge=1, le=500, description="Number of questions to generate")
    question_types: list[Literal["mcq", "short", "long"]] = Field(
        default=["mcq", "short"],
        description="Types of questions to generate"
    )
    chapters: list[str] | None = Field(
        default=None,
        description="Specific chapters to focus on (optional, if None uses all chapters)"
    )


class AdminGenerateQuestionsResponse(BaseModel):
    """Response for admin question generation request."""
    job_id: str
    status: str = "queued"
    message: str = "Question generation job has been queued. Check job status via /api/v1/jobs/{job_id}"


class AdminBatchGenerateQuestionsRequest(BaseModel):
    """
    Batch admin-triggered question generation request for multiple PDFs.
    """
    document_ids: list[str] = Field(..., min_length=1, description="List of document IDs")
    from_class: int = Field(..., ge=1, le=12, description="Starting class level")
    to_class: int = Field(..., ge=1, le=12, description="Ending class level (inclusive)")
    subject: str = Field(..., description="Subject name")
    difficulty: Literal["easy", "medium", "hard", "any"] = Field(
        default="any",
        description="Difficulty level ('any' generates for all difficulty levels)"
    )
    question_count: int = Field(..., ge=1, le=500, description="Number of questions to generate per document")
    question_types: list[Literal["mcq", "descriptive", "short", "long"]] = Field(
        default=["mcq", "descriptive"],
        description="Types of questions to generate ('descriptive' maps to 'short' and 'long')"
    )
    chapters: list[str] | None = Field(
        default=None,
        description="Specific chapters to focus on (optional, if None uses all chapters)"
    )


class DocumentJobStatus(BaseModel):
    """Status of a single document's generation job."""
    document_id: str
    job_id: str | None = None
    status: str = "queued"  # queued, claimed, running, completed, completed_partial, failed_retryable, failed_terminal
    message: str = ""
    error: str | None = None


class AdminBatchGenerateQuestionsResponse(BaseModel):
    """Response for batch admin question generation request."""
    total_documents: int
    jobs_created: int
    documents: list[DocumentJobStatus]
    message: str = "Batch question generation jobs have been queued"


class AdminReindexRequest(BaseModel):
    """Request to re-index existing documents."""
    document_ids: list[str] = Field(..., min_length=1, description="List of document IDs to re-index")


class AdminReindexResponse(BaseModel):
    """Response for admin re-index request."""
    job_id: str
    status: str = "queued"
    documents: int
    message: str = "Re-index job has been queued"


class AdminPrewarmPoolsResponse(BaseModel):
    """Response for global pool prewarm."""
    status: str = "success"
    prewarm_id: str
    job_id: str
    questions_promoted: int
    pools_created: int
    promoted_question_ids: list[str] = Field(default_factory=list)


class AdminPrewarmQuestionOut(BaseModel):
    """Canonical promoted question payload for Niyukti persistence."""

    question_id: str
    board: str
    class_level: int
    subject: str
    syllabus_version: str
    chapter_number: int
    difficulty: str
    question_type: str
    question_json: dict
    source_document_id: str
    rag_generation_job_id: str


class AdminPrewarmQuestionsResponse(BaseModel):
    """Promoted questions from a specific prewarm run."""

    prewarm_id: str
    questions: list[AdminPrewarmQuestionOut]


class AdminGeneratedQuestionOut(BaseModel):
    id: str
    class_level: int
    subject: str
    chapter: str
    chapter_number: int | None = None
    chapter_name: str | None = None
    difficulty: str
    qtype: str
    language: str
    question_json: dict
    created_at: datetime


class AdminGeneratedQuestionsResponse(BaseModel):
    questions: list[AdminGeneratedQuestionOut] = Field(default_factory=list)

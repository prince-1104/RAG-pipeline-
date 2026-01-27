from __future__ import annotations

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
    status: str = "queued"  # queued, running, succeeded, failed
    message: str = ""
    error: str | None = None


class AdminBatchGenerateQuestionsResponse(BaseModel):
    """Response for batch admin question generation request."""
    total_documents: int
    jobs_created: int
    documents: list[DocumentJobStatus]
    message: str = "Batch question generation jobs have been queued"

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class Envelope(BaseModel):
    request_id: str | None = None


class JobRef(BaseModel):
    job_id: str
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    storage_uri: str
    page_start: int
    page_end: int
    excerpt: str = Field(..., description="Short quoted excerpt from retrieved context used for the answer.")



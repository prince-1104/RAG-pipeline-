from __future__ import annotations

from pydantic import BaseModel, Field


class CorpusCreateRequest(BaseModel):
    tenant_id: str = Field(..., description="Niyukti tenant/institute identifier (opaque to this service).")
    name: str
    description: str | None = None


class CorpusResponse(BaseModel):
    corpus_id: str
    name: str
    board: str
    tenant_id: str


class DocumentCreateRequest(BaseModel):
    """
    PDF upload metadata - REQUIRED fields for question pool system.
    All fields are mandatory to ensure proper pool generation.
    """
    syllabus: str = Field(..., description="Syllabus identifier, e.g. 'CBSE'", examples=["CBSE"])
    class_level: int = Field(..., ge=1, le=12, description="Class level (1-12)")
    subject: str = Field(..., examples=["Science", "Mathematics", "English"])
    filename: str
    syllabus_version: str = Field(default="2024", description="Syllabus version, e.g. '2024'")
    
    # Optional but preferred for better chunking
    chapter: str | None = Field(
        default=None,
        description="Chapter name (optional but preferred for better question pool organization)",
    )

    # optional: to improve syllabus-aware chunking
    toc: list[dict] | None = Field(
        default=None,
        description="Optional table-of-contents structure extracted/curated upstream. Each entry should include title, page_start, page_end.",
    )


class DocumentRegisterRequest(BaseModel):
    """
    Register an existing local PDF (no upload).
    For local PDF mode: PDF must exist under PDF_BASE_PATH/cbse/{class}/{subject}/{filename}
    
    REQUIRED fields for question pool system.
    """
    syllabus: str = Field(..., description="Syllabus identifier, e.g. 'CBSE'", examples=["CBSE"])
    class_level: int = Field(..., ge=1, le=12, description="Class level (1-12)")
    subject: str = Field(..., examples=["Science", "Mathematics", "English"])
    filename: str = Field(..., description="Filename relative to PDF_BASE_PATH/cbse/{class}/{subject}/")
    syllabus_version: str = Field(default="2024", description="Syllabus version, e.g. '2024'")
    
    # Optional but preferred
    chapter: str | None = Field(
        default=None,
        description="Chapter name (optional but preferred)",
    )

    # optional: to improve syllabus-aware chunking
    toc: list[dict] | None = Field(
        default=None,
        description="Optional table-of-contents structure extracted/curated upstream. Each entry should include title, page_start, page_end.",
    )


class DocumentResponse(BaseModel):
    document_id: str
    corpus_id: str
    filename: str
    storage_uri: str
    syllabus: str | None = None
    class_level: int | None
    subject: str | None
    syllabus_version: str | None = None
    chapter: str | None = None


class IndexJobRequest(BaseModel):
    tenant_id: str
    document_ids: list[str] | None = Field(default=None, description="If omitted, indexes all documents in the corpus.")


class IndexJobResponse(BaseModel):
    job_id: str
    status: str



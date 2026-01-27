from __future__ import annotations

from pydantic import BaseModel, Field


class PDFIngestRequest(BaseModel):
    """Metadata for PDF ingestion."""
    doc_id: str = Field(..., description="Document ID provided by caller")
    board: str = Field(..., description="Board identifier (e.g., 'CBSE')")
    class_level: int = Field(..., ge=1, le=12, description="Class level (1-12)")
    subject: str = Field(..., description="Subject name")


class PDFIngestResponse(BaseModel):
    """Response for PDF ingestion."""
    success: bool
    doc_id: str
    corpus_id: str | None = None
    storage_uri: str | None = None
    job_id: str | None = None  # Ingest job ID
    index_job_id: str | None = None  # Index job ID (for tracking indexing progress)
    document_id: str | None = None  # Document ID in database
    error: str | None = None

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Enum as SQLEnum
from sqlmodel import Field, SQLModel, Column


class JobKind(str, enum.Enum):
    """Job kind enum - values must match database enum exactly (lowercase)."""
    ingest = "ingest"
    index = "index"
    generate = "generate"
    generate_question_pool = "generate_question_pool"
    refresh_question_pool = "refresh_question_pool"
    
    def __str__(self) -> str:
        """Ensure str() returns the enum value, not the member name."""
        return self.value


class JobStatus(str, enum.Enum):
    """Job status enum - values must match database enum exactly (lowercase)."""
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    
    def __str__(self) -> str:
        """Ensure str() returns the enum value, not the member name."""
        return self.value


class Corpus(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

    name: str
    board: str = "CBSE"
    description: str | None = None

    # multi-tenant integration boundary: Niyukti provides an external institute/tenant identifier.
    tenant_id: str

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    corpus_id: str = Field(index=True, foreign_key="corpus.id")

    filename: str
    storage_uri: str  # s3://bucket/key or file://...

    # REQUIRED metadata for question pool system
    syllabus: str = Field(index=True, default="CBSE")  # e.g., "CBSE"
    class_level: int | None = Field(default=None, index=True)
    subject: str | None = Field(default=None, index=True)
    syllabus_version: str = Field(index=True, default="2024")  # e.g., "2024"
    chapter: str | None = Field(default=None, index=True)  # Optional but preferred

    # optional hints to help syllabus-aware chunking
    toc_json: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    corpus_id: str = Field(index=True, foreign_key="corpus.id")
    document_id: str = Field(index=True, foreign_key="document.id")

    # syllabus-aware tags used for filtering BEFORE vector search
    class_level: int = Field(index=True)
    subject: str = Field(index=True)
    chapter: str = Field(index=True)
    topics_json: str | None = None

    page_start: int = Field(index=True)
    page_end: int = Field(index=True)

    text: str
    text_hash: str = Field(index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class GenerationJob(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    kind: JobKind = Field(index=True)
    status: JobStatus = Field(index=True, default=JobStatus.queued)

    tenant_id: str = Field(index=True)
    corpus_id: str | None = Field(default=None, index=True)

    # request + result payloads (JSON strings to keep schema evolution easy)
    request_json: str
    result_json: str | None = None

    error_code: str | None = None
    error_message: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class GeneratedQuestion(SQLModel, table=True):
    """
    Legacy table: Kept for backward compatibility.
    New questions should use PoolQuestion + JobQuestion mapping.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    job_id: str = Field(index=True, foreign_key="generationjob.id")
    corpus_id: str = Field(index=True, foreign_key="corpus.id")

    class_level: int = Field(index=True)
    subject: str = Field(index=True)
    chapter: str = Field(index=True)
    difficulty: str = Field(index=True)
    qtype: str = Field(index=True)
    language: str = Field(index=True, default="en")

    question_json: str  # strict schema with citations
    dedupe_key: str = Field(index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionCitation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    question_id: str = Field(index=True, foreign_key="generatedquestion.id")
    chunk_id: str = Field(index=True, foreign_key="chunk.id")
    document_id: str = Field(index=True, foreign_key="document.id")

    storage_uri: str
    page_start: int
    page_end: int
    excerpt: str

    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionPoolStatus(str, enum.Enum):
    pending = "pending"
    generating = "generating"
    ready = "ready"
    failed = "failed"


class QuestionPool(SQLModel, table=True):
    """
    Question Pool: Pre-generated questions for a specific syllabus/class/subject/difficulty combination.
    Pools are shared across institutes and regenerated only on syllabus changes.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    
    # Pool identity (unique constraint)
    syllabus: str = Field(index=True)  # e.g., "CBSE"
    class_level: int = Field(index=True)
    subject: str = Field(index=True)
    difficulty: str = Field(index=True)  # easy | medium | hard
    syllabus_version: str = Field(index=True, default="2024")  # e.g., "2024"
    
    status: QuestionPoolStatus = Field(index=True, default=QuestionPoolStatus.pending)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PoolQuestion(SQLModel, table=True):
    """
    Questions stored in pools (separate from GeneratedQuestion for clarity).
    PoolQuestions are reusable across jobs.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    pool_id: str = Field(index=True, foreign_key="questionpool.id")
    
    question_text: str  # stem
    options: str  # JSON array of options
    correct_answer: str | None = None
    concept: str | None = None  # Examined concept
    difficulty: str = Field(index=True)
    source_chunk_id: str | None = Field(index=True, foreign_key="chunk.id")
    
    # Full question JSON for compatibility
    question_json: str
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobQuestion(SQLModel, table=True):
    """
    Mapping table: links jobs to questions from pools.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    job_id: str = Field(index=True, foreign_key="generationjob.id")
    question_id: str = Field(index=True, foreign_key="poolquestion.id")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)



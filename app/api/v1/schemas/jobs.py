from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    job_id: str
    kind: Literal[
        "ingest",
        "index",
        "generate",
        "generate_question_pool",
        "refresh_question_pool",
        "admin_generate_questions",
    ]
    status: Literal[
        "queued",
        "claimed",
        "running",
        "completed",
        "completed_partial",
        "failed_retryable",
        "failed_terminal",
    ]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None  # Final result (when completed) or progress (when running)



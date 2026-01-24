from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    job_id: str
    kind: Literal["INGEST", "INDEX", "GENERATE"]
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None  # Final result (when SUCCEEDED) or progress (when RUNNING)



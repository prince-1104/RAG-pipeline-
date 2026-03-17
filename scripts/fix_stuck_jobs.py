"""
One-time script to mark stuck jobs (e.g. null-byte index failures) as failed_terminal
so they stop retrying forever. Run from project root: python -m scripts.fix_stuck_jobs
"""
from __future__ import annotations

import asyncio
import os
import sys

# Ensure app is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, update

from app.db.models import GenerationJob, JobStatus
from app.db.session import get_session_ctx


async def fix_stuck_jobs() -> None:
    async with get_session_ctx() as session:
        # Kill the specific stuck job from logs
        await session.execute(
            update(GenerationJob)
            .where(GenerationJob.id == "8db793fd-df4e-4e87-8ae2-d6fc32e18223")
            .values(
                status=JobStatus.failed_terminal,
                error_message="Killed: null byte encoding error — re-upload PDF after fix",
            )
        )

        # Also kill any other jobs stuck in failed_retryable with retry_count >= 3
        result = await session.execute(
            select(GenerationJob).where(
                GenerationJob.status == JobStatus.failed_retryable,
                GenerationJob.retry_count >= 3,
            )
        )
        stuck_jobs = result.scalars().all()
        for job in stuck_jobs:
            job.status = JobStatus.failed_terminal
            job.error_message = "Killed: exceeded max retries"
            session.add(job)

        await session.commit()
        print(f"Fixed 1 + {len(stuck_jobs)} stuck job(s)")


if __name__ == "__main__":
    asyncio.run(fix_stuck_jobs())

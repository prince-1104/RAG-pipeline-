from __future__ import annotations

"""
Run `update_job_statuses.sql` safely without `psql`.

Why this exists:
- Postgres enum `ALTER TYPE ... ADD VALUE` requires a commit boundary before the
  new enum labels can be used. If you execute the entire SQL file as one query,
  you can hit: psycopg.errors.UnsafeNewEnumValueUsage

This runner:
- Reads migrations/update_job_statuses.sql
- Splits it into top-level SQL statements while preserving DO $$ ... $$ blocks
- Executes statements one-by-one with autocommit on
- Prints the final enum labels for verification
"""

import sys
from pathlib import Path

import psycopg

# Ensure repo root is on sys.path when running from migrations/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import get_settings  # noqa: E402


def _extract_do_blocks(sql: str) -> tuple[list[str], str]:
    """
    Extract `DO $$ ... END $$;` blocks (including all internal semicolons).
    Returns (do_blocks, remainder_sql_without_do_blocks).

    This migration file is simple and uses only `DO $$` blocks; line-based
    extraction is more reliable than a full SQL parser here.
    """
    do_blocks: list[str] = []
    remainder: list[str] = []

    in_do = False
    buf: list[str] = []

    for line in sql.splitlines(keepends=True):
        if not in_do and line.lstrip().startswith("DO $$"):
            in_do = True
            buf = [line]
            continue
        if in_do:
            buf.append(line)
            if line.strip() == "END $$;":
                do_blocks.append("".join(buf))
                buf = []
                in_do = False
            continue
        remainder.append(line)

    # If file ended mid-block, treat it as remainder (best-effort)
    if buf:
        remainder.extend(buf)

    return do_blocks, "".join(remainder)


def main() -> None:
    settings = get_settings()
    sql_path = Path("migrations/update_job_statuses.sql")
    sql = sql_path.read_text(encoding="utf-8")

    do_blocks, remainder = _extract_do_blocks(sql)
    remainder_statements = [s.strip() for s in remainder.split(";") if s.strip()]

    # Phase 1: add enum values via DO blocks, committing between blocks.
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            for blk in do_blocks:
                cur.execute(blk)
                conn.commit()

    # Phase 2: run the rest. We start a NEW session to avoid
    # UnsafeNewEnumValueUsage when using newly-added enum labels.
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            for stmt in remainder_statements:
                cur.execute(stmt)
                conn.commit()

            cur.execute(
                "SELECT enumlabel FROM pg_enum "
                "WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus') "
                "ORDER BY enumsortorder;"
            )
            labels = [r[0] for r in (cur.fetchall() or [])]
            print("jobstatus_enum:", labels)


if __name__ == "__main__":
    main()


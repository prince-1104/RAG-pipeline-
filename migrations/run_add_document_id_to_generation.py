from __future__ import annotations

"""
Run add_document_id_to_generation.sql safely without psql.
Executes DO $$ blocks first, then remaining statements.
"""

from pathlib import Path
import sys

import psycopg

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import get_settings  # noqa: E402


def _extract_do_blocks(sql: str) -> tuple[list[str], str]:
    blocks: list[str] = []
    remainder: list[str] = []
    in_do = False
    buf: list[str] = []

    for line in sql.splitlines(keepends=True):
        if (not in_do) and line.lstrip().startswith("DO $$"):
            in_do = True
            buf = [line]
            continue
        if in_do:
            buf.append(line)
            if line.strip() == "END $$;":
                blocks.append("".join(buf))
                buf = []
                in_do = False
            continue
        remainder.append(line)

    if buf:
        remainder.extend(buf)

    return blocks, "".join(remainder)


def main() -> None:
    sql_path = Path("migrations/add_document_id_to_generation.sql")
    sql = sql_path.read_text(encoding="utf-8")
    blocks, remainder = _extract_do_blocks(sql)
    statements = [s.strip() for s in remainder.split(";") if s.strip()]

    url = get_settings().database_url

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for blk in blocks:
                cur.execute(blk)
                conn.commit()

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
                conn.commit()

    print("migration add_document_id_to_generation.sql applied")


if __name__ == "__main__":
    main()

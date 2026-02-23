from __future__ import annotations

"""
Run add_curriculum_metadata_fields.sql safely without psql.
"""

from pathlib import Path
import sys

import psycopg

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import get_settings  # noqa: E402


def main() -> None:
    sql_path = Path("migrations/add_curriculum_metadata_fields.sql")
    sql = sql_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    url = get_settings().database_url

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
                conn.commit()

    print("migration add_curriculum_metadata_fields.sql applied")


if __name__ == "__main__":
    main()

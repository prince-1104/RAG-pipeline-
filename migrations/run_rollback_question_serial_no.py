from __future__ import annotations

"""
Run rollback_question_serial_no.sql to remove serial_no column and sequence.
Only use if you previously applied add_question_serial_no and want to undo DB changes.
"""

from pathlib import Path
import sys

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.settings import get_settings  # noqa: E402


def main() -> None:
    sql_path = Path(__file__).resolve().parent / "rollback_question_serial_no.sql"
    sql = sql_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    url = get_settings().database_url

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
                conn.commit()

    print("rollback_question_serial_no.sql applied: serial_no and question_serial_seq removed.")


if __name__ == "__main__":
    main()

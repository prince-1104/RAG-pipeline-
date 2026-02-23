"""
Database migration utilities.
Run migrations to update schema when models change.
"""
from __future__ import annotations

import re

from sqlalchemy import text
from sqlmodel import Session

from app.db.session import get_engine, get_session
from app.observability.logging import log


def migrate_add_document_syllabus_fields() -> None:
    """
    Add syllabus, syllabus_version, and chapter columns to document table if they don't exist.
    This migration is idempotent - safe to run multiple times.
    """
    engine = get_engine()
    
    with Session(engine) as session:
        try:
            # Check if columns exist
            result = session.exec(
                text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'document' 
                    AND column_name IN ('syllabus', 'syllabus_version', 'chapter')
                """)
            ).all()
            
            existing_columns = {row[0] for row in result}
            
            # Add syllabus column if missing
            if 'syllabus' not in existing_columns:
                log.info("migration_adding_column", table="document", column="syllabus")
                session.exec(
                    text("""
                        ALTER TABLE document 
                        ADD COLUMN syllabus VARCHAR NOT NULL DEFAULT 'CBSE'
                    """)
                )
                session.exec(
                    text("CREATE INDEX IF NOT EXISTS ix_document_syllabus ON document(syllabus)")
                )
                session.commit()
                log.info("migration_column_added", table="document", column="syllabus")
            else:
                log.info("migration_column_exists", table="document", column="syllabus")
            
            # Add syllabus_version column if missing
            if 'syllabus_version' not in existing_columns:
                log.info("migration_adding_column", table="document", column="syllabus_version")
                session.exec(
                    text("""
                        ALTER TABLE document 
                        ADD COLUMN syllabus_version VARCHAR NOT NULL DEFAULT '2024'
                    """)
                )
                session.exec(
                    text("CREATE INDEX IF NOT EXISTS ix_document_syllabus_version ON document(syllabus_version)")
                )
                session.commit()
                log.info("migration_column_added", table="document", column="syllabus_version")
            else:
                log.info("migration_column_exists", table="document", column="syllabus_version")
            
            # Add chapter column if missing (nullable since it's optional)
            if 'chapter' not in existing_columns:
                log.info("migration_adding_column", table="document", column="chapter")
                session.exec(
                    text("""
                        ALTER TABLE document 
                        ADD COLUMN chapter VARCHAR
                    """)
                )
                session.exec(
                    text("CREATE INDEX IF NOT EXISTS ix_document_chapter ON document(chapter)")
                )
                session.commit()
                log.info("migration_column_added", table="document", column="chapter")
            else:
                log.info("migration_column_exists", table="document", column="chapter")
            
            log.info("migration_complete", migration="add_document_syllabus_fields")
            
        except Exception as e:
            session.rollback()
            log.exception("migration_error", error=str(e))
            raise


def migrate_add_job_kind_admin_generate_questions() -> None:
    """
    Add 'admin_generate_questions' to the JobKind enum if it doesn't exist.
    Note: This assumes PostgreSQL enum type. For other databases, enum handling differs.
    """
    engine = get_engine()
    
    # Only run for PostgreSQL
    if not engine.url.drivername.startswith("postgresql"):
        log.info("migration_skipped", reason="not_postgresql", migration="add_job_kind_enum")
        return
    
    with Session(engine) as session:
        try:
            # Check if enum value exists
            result = session.exec(
                text("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_enum 
                        WHERE enumlabel = 'admin_generate_questions'
                        AND enumtypid = (
                            SELECT oid 
                            FROM pg_type 
                            WHERE typname = 'jobkind'
                        )
                    )
                """)
            ).first()
            
            if not result[0]:
                log.info("migration_adding_enum_value", enum="JobKind", value="admin_generate_questions")
                session.exec(
                    text("ALTER TYPE jobkind ADD VALUE IF NOT EXISTS 'admin_generate_questions'")
                )
                session.commit()
                log.info("migration_enum_value_added", enum="JobKind", value="admin_generate_questions")
            else:
                log.info("migration_enum_value_exists", enum="JobKind", value="admin_generate_questions")
            
        except Exception as e:
            # If enum doesn't exist as a PostgreSQL type (e.g., using VARCHAR), skip
            if "does not exist" in str(e).lower() or "type" in str(e).lower():
                log.info("migration_skipped", reason="enum_not_postgres_type", migration="add_job_kind_enum")
            else:
                session.rollback()
                log.exception("migration_error", error=str(e))
                raise


def migrate_add_question_pool_status_ready_partial() -> None:
    """
    Add 'ready_partial' to the QuestionPoolStatus enum if it doesn't exist.
    Only needed when QuestionPoolStatus is backed by a PostgreSQL enum type.
    """
    engine = get_engine()
    if not engine.url.drivername.startswith("postgresql"):
        log.info("migration_skipped", reason="not_postgresql", migration="add_question_pool_status_enum")
        return

    with Session(engine) as session:
        try:
            result = session.exec(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_enum
                        WHERE enumlabel = 'ready_partial'
                        AND enumtypid = (
                            SELECT oid
                            FROM pg_type
                            WHERE typname = 'questionpoolstatus'
                        )
                    )
                    """
                )
            ).first()

            if result and not result[0]:
                log.info("migration_adding_enum_value", enum="QuestionPoolStatus", value="ready_partial")
                session.exec(text("ALTER TYPE questionpoolstatus ADD VALUE IF NOT EXISTS 'ready_partial'"))
                session.commit()
                log.info("migration_enum_value_added", enum="QuestionPoolStatus", value="ready_partial")
            else:
                log.info("migration_enum_value_exists", enum="QuestionPoolStatus", value="ready_partial")
        except Exception as e:
            # If enum doesn't exist as a PostgreSQL type (e.g., using VARCHAR), skip
            if "does not exist" in str(e).lower() or "type" in str(e).lower():
                log.info("migration_skipped", reason="enum_not_postgres_type", migration="add_question_pool_status_enum")
            else:
                session.rollback()
                log.exception("migration_error", error=str(e))
                raise


def migrate_add_question_pool_bookkeeping_fields() -> None:
    """
    Add requested_count, generated_count, last_error columns to questionpool table if missing.
    Idempotent.
    """
    engine = get_engine()
    with Session(engine) as session:
        try:
            result = session.exec(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'questionpool'
                    AND column_name IN ('requested_count', 'generated_count', 'last_error')
                    """
                )
            ).all()
            existing_columns = {row[0] for row in result}

            if "requested_count" not in existing_columns:
                session.exec(text("ALTER TABLE questionpool ADD COLUMN requested_count INTEGER"))
            if "generated_count" not in existing_columns:
                session.exec(text("ALTER TABLE questionpool ADD COLUMN generated_count INTEGER"))
            if "last_error" not in existing_columns:
                session.exec(text("ALTER TABLE questionpool ADD COLUMN last_error VARCHAR"))

            session.commit()
            log.info("migration_complete", migration="add_question_pool_bookkeeping_fields")
        except Exception as e:
            session.rollback()
            log.exception("migration_error", error=str(e))
            raise


def migrate_add_question_pool_chapter() -> None:
    """
    Add chapter column to questionpool table if missing.
    Idempotent.
    """
    engine = get_engine()
    with Session(engine) as session:
        try:
            result = session.exec(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'questionpool'
                    AND column_name = 'chapter'
                    """
                )
            ).all()
            existing_columns = {row[0] for row in result}

            if "chapter" not in existing_columns:
                session.exec(
                    text(
                        """
                        ALTER TABLE questionpool
                        ADD COLUMN chapter VARCHAR NOT NULL DEFAULT 'Unknown Chapter'
                        """
                    )
                )
                session.exec(text("CREATE INDEX IF NOT EXISTS ix_questionpool_chapter ON questionpool(chapter)"))
                session.commit()
                log.info("migration_column_added", table="questionpool", column="chapter")
            else:
                log.info("migration_column_exists", table="questionpool", column="chapter")
        except Exception as e:
            session.rollback()
            log.exception("migration_error", error=str(e))
            raise


def _parse_generated_question_chapter(chapter: str | None) -> tuple[int | None, str | None]:
    if not chapter:
        return None, None
    value = str(chapter).strip()
    if not value:
        return None, None
    if value.isdigit():
        return int(value), None

    match = re.match(
        r"^(?:chapter|unit|ch)\s*(\d+)\s*[:\-]\s*(.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), (match.group(2) or "").strip() or None

    match = re.match(r"^(?:chapter|unit|ch)\s*(\d+)\s*$", value, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), None

    return None, value


def migrate_add_generatedquestion_chapter_identity() -> None:
    """
    Add chapter_number and chapter_name columns to generatedquestion table if missing.
    Also backfill best-effort values from existing chapter string.
    Idempotent.
    """
    engine = get_engine()
    with Session(engine) as session:
        try:
            result = session.exec(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'generatedquestion'
                    AND column_name IN ('chapter_number', 'chapter_name')
                    """
                )
            ).all()
            existing_columns = {row[0] for row in result}

            if "chapter_number" not in existing_columns:
                session.exec(text("ALTER TABLE generatedquestion ADD COLUMN chapter_number INTEGER"))
                session.exec(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_generatedquestion_chapter_number "
                        "ON generatedquestion(chapter_number)"
                    )
                )
                log.info("migration_column_added", table="generatedquestion", column="chapter_number")
            else:
                log.info("migration_column_exists", table="generatedquestion", column="chapter_number")

            if "chapter_name" not in existing_columns:
                session.exec(text("ALTER TABLE generatedquestion ADD COLUMN chapter_name VARCHAR"))
                session.exec(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_generatedquestion_chapter_name "
                        "ON generatedquestion(chapter_name)"
                    )
                )
                log.info("migration_column_added", table="generatedquestion", column="chapter_name")
            else:
                log.info("migration_column_exists", table="generatedquestion", column="chapter_name")

            session.commit()

            rows = session.exec(
                text(
                    """
                    SELECT id, chapter, chapter_number, chapter_name
                    FROM generatedquestion
                    """
                )
            ).all()
            for row in rows:
                row_id = str(row[0])
                chapter = row[1]
                existing_number = row[2]
                existing_name = row[3]
                parsed_number, parsed_name = _parse_generated_question_chapter(chapter)
                next_number = existing_number if existing_number is not None else parsed_number
                next_name = existing_name if existing_name else parsed_name
                if next_number == existing_number and next_name == existing_name:
                    continue
                session.execute(
                    text(
                        """
                        UPDATE generatedquestion
                        SET chapter_number = :chapter_number,
                            chapter_name = :chapter_name
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": row_id,
                        "chapter_number": next_number,
                        "chapter_name": next_name,
                    },
                )
            session.commit()
            log.info("migration_complete", migration="add_generatedquestion_chapter_identity")
        except Exception as e:
            session.rollback()
            log.exception("migration_error", error=str(e))
            raise


def run_all_migrations() -> None:
    """Run all pending migrations."""
    log.info("migrations_starting")
    migrate_add_document_syllabus_fields()
    migrate_add_job_kind_admin_generate_questions()
    migrate_add_question_pool_status_ready_partial()
    migrate_add_question_pool_bookkeeping_fields()
    migrate_add_question_pool_chapter()
    migrate_add_generatedquestion_chapter_identity()
    log.info("migrations_complete")


if __name__ == "__main__":
    run_all_migrations()

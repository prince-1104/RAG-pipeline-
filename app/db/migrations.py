"""
Database migration utilities.
Run migrations to update schema when models change.
"""
from __future__ import annotations

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


def run_all_migrations() -> None:
    """Run all pending migrations."""
    log.info("migrations_starting")
    migrate_add_document_syllabus_fields()
    migrate_add_job_kind_admin_generate_questions()
    log.info("migrations_complete")


if __name__ == "__main__":
    run_all_migrations()

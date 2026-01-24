"""
Database migration: Add question pool tables and update Document schema.

Run this migration before deploying the refactored system.

Usage:
    python migrations/add_question_pools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlmodel import SQLModel, create_engine

from app.db.models import (
    Document,
    JobQuestion,
    PoolQuestion,
    QuestionPool,
    QuestionPoolStatus,
)
from app.settings import get_settings


def run_migration():
    """Create new tables and add columns to existing tables."""
    settings = get_settings()
    
    # Convert postgresql:// to postgresql+psycopg:// for psycopg3
    db_url = settings.database_url
    if db_url.startswith("postgresql://") and not db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    engine = create_engine(db_url, echo=True)
    
    print("Creating new tables: QuestionPool, PoolQuestion, JobQuestion...")
    SQLModel.metadata.create_all(engine, tables=[
        QuestionPool.__table__,
        PoolQuestion.__table__,
        JobQuestion.__table__,
    ])
    
    # Add columns to existing Document table if they don't exist
    print("\nUpdating Document table schema...")
    with engine.connect() as conn:
        # Check and add syllabus column
        try:
            conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS syllabus VARCHAR"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_syllabus ON document(syllabus)"))
            conn.execute(text("UPDATE document SET syllabus = 'CBSE' WHERE syllabus IS NULL"))
            print("✓ Added syllabus column to Document table")
        except Exception as e:
            print(f"⚠ Warning adding syllabus column: {e}")
        
        # Check and add syllabus_version column
        try:
            conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS syllabus_version VARCHAR"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_syllabus_version ON document(syllabus_version)"))
            conn.execute(text("UPDATE document SET syllabus_version = '2024' WHERE syllabus_version IS NULL"))
            print("✓ Added syllabus_version column to Document table")
        except Exception as e:
            print(f"⚠ Warning adding syllabus_version column: {e}")
        
        # Check and add chapter column
        try:
            conn.execute(text("ALTER TABLE document ADD COLUMN IF NOT EXISTS chapter VARCHAR"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_document_chapter ON document(chapter)"))
            print("✓ Added chapter column to Document table")
        except Exception as e:
            print(f"⚠ Warning adding chapter column: {e}")
        
        conn.commit()
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Verify Document records have syllabus and syllabus_version set")
    print("2. Run indexing jobs to ensure chunks have proper metadata")
    print("3. Start Redis server: redis-server")
    print("4. Start background workers: python -m app.worker")
    print("5. Test pool generation via API endpoint")


if __name__ == "__main__":
    run_migration()

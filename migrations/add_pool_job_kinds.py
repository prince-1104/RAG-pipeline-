"""
Migration: Add GENERATE_QUESTION_POOL and REFRESH_QUESTION_POOL to JobKind enum.

Run this script to update the PostgreSQL enum type.

Usage:
    python migrations/add_pool_job_kinds.py
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import get_engine
from sqlalchemy import text


def run_migration():
    """Add new enum values to jobkind enum."""
    try:
        engine = get_engine()
        
        with engine.begin() as conn:  # Use begin() for transaction management
            # Add GENERATE_QUESTION_POOL using DO block (PostgreSQL doesn't support IF NOT EXISTS directly)
            add_generate_pool = text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_enum 
                        WHERE enumlabel = 'GENERATE_QUESTION_POOL' 
                        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                    ) THEN
                        ALTER TYPE jobkind ADD VALUE 'GENERATE_QUESTION_POOL';
                    END IF;
                END $$;
            """)
            conn.execute(add_generate_pool)
            print("✅ Added GENERATE_QUESTION_POOL to jobkind enum")
            
            # Add REFRESH_QUESTION_POOL
            add_refresh_pool = text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_enum 
                        WHERE enumlabel = 'REFRESH_QUESTION_POOL' 
                        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                    ) THEN
                        ALTER TYPE jobkind ADD VALUE 'REFRESH_QUESTION_POOL';
                    END IF;
                END $$;
            """)
            conn.execute(add_refresh_pool)
            print("✅ Added REFRESH_QUESTION_POOL to jobkind enum")
            
            # Verify enum values
            verify_query = text("""
                SELECT enumlabel FROM pg_enum 
                WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                ORDER BY enumsortorder;
            """)
            result = conn.execute(verify_query)
            enum_values = [row[0] for row in result]
            
            print("\n📋 Current jobkind enum values:")
            for value in enum_values:
                print(f"   - {value}")
            
            print("\n✅ Migration complete!")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\n💡 Alternative: Run the SQL script directly:")
        print("   migrations/add_pool_job_kinds.sql")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()

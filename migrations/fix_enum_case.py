"""
Migration: Fix enum case mismatch - change to lowercase to match SQLAlchemy serialization.

Run this script to update the PostgreSQL enum type to use lowercase values.
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import get_engine
from sqlalchemy import text


def run_migration():
    """Add lowercase enum values and update existing data."""
    try:
        engine = get_engine()
        
        # Step 1: Add lowercase enum values (must commit before using)
        print("Step 1: Adding lowercase enum values...")
        with engine.begin() as conn:
            add_lowercase_generate = text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_enum 
                        WHERE enumlabel = 'generate_question_pool' 
                        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                    ) THEN
                        ALTER TYPE jobkind ADD VALUE 'generate_question_pool';
                    END IF;
                END $$;
            """)
            conn.execute(add_lowercase_generate)
            print("[OK] Added generate_question_pool (lowercase) to jobkind enum")
            
            add_lowercase_refresh = text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_enum 
                        WHERE enumlabel = 'refresh_question_pool' 
                        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                    ) THEN
                        ALTER TYPE jobkind ADD VALUE 'refresh_question_pool';
                    END IF;
                END $$;
            """)
            conn.execute(add_lowercase_refresh)
            print("[OK] Added refresh_question_pool (lowercase) to jobkind enum")
        # Transaction commits here
        
        # Step 2: Update existing records (in NEW transaction after enum values are committed)
        print("\nStep 2: Updating existing records...")
        with engine.begin() as conn:
            update_generate = text("""
                UPDATE generationjob 
                SET kind = 'generate_question_pool'::jobkind 
                WHERE kind::text = 'GENERATE_QUESTION_POOL'
            """)
            result1 = conn.execute(update_generate)
            print(f"[OK] Updated {result1.rowcount} records from GENERATE_QUESTION_POOL to generate_question_pool")
            
            update_refresh = text("""
                UPDATE generationjob 
                SET kind = 'refresh_question_pool'::jobkind 
                WHERE kind::text = 'REFRESH_QUESTION_POOL'
            """)
            result2 = conn.execute(update_refresh)
            print(f"[OK] Updated {result2.rowcount} records from REFRESH_QUESTION_POOL to refresh_question_pool")
        # Transaction commits here
        
        # Step 3: Verify enum values
        print("\nStep 3: Verifying enum values...")
        with engine.connect() as conn:
            verify_query = text("""
                SELECT enumlabel FROM pg_enum 
                WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
                ORDER BY enumsortorder;
            """)
            result = conn.execute(verify_query)
            enum_values = [row[0] for row in result]
            
            print("\nCurrent jobkind enum values:")
            for value in enum_values:
                print(f"   - {value}")
            
            print("\n[OK] Migration complete!")
            print("\n[NOTE] Uppercase enum values (GENERATE_QUESTION_POOL, REFRESH_QUESTION_POOL) still exist.")
            print("       They are safe to leave - PostgreSQL will use lowercase versions for new records.")
            
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        print("\n[INFO] Alternative: Run the SQL script directly:")
        print("   migrations/fix_enum_case.sql")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_migration()

# Database Migration: Add Pool Job Kinds

## Problem
The database enum `jobkind` doesn't have the values `GENERATE_QUESTION_POOL` and `REFRESH_QUESTION_POOL`, causing errors when trying to create pool generation jobs.

## Solution
Run the SQL migration script to add the missing enum values.

## Option 1: Using psql (Recommended)

```bash
# Connect to your database
psql "postgresql://neondb_owner:npg_E5zrfgZYJ3yx@ep-quiet-fog-a427wugk-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Run the migration
\i migrations/add_pool_job_kinds.sql
```

## Option 2: Using Python Script

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run migration script
python migrations/add_pool_job_kinds.py
```

## Option 3: Direct SQL Execution

Copy the SQL from `migrations/add_pool_job_kinds.sql` and run it directly in your database client (pgAdmin, DBeaver, etc.).

## Verification

After running the migration, verify the enum values exist:

```sql
SELECT enumlabel FROM pg_enum 
WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
ORDER BY enumsortorder;
```

You should see:
- INGEST
- INDEX
- GENERATE
- GENERATE_QUESTION_POOL ✅
- REFRESH_QUESTION_POOL ✅

## After Migration

Restart your FastAPI server and worker. The pool generation jobs should now work correctly.

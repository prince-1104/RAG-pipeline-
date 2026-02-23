-- Migration: Update job status enum + add retry/claim fields
-- Run this SQL script directly on your PostgreSQL database

-- Add new enum values if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'claimed'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
    ) THEN
        ALTER TYPE jobstatus ADD VALUE 'claimed';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'completed'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
    ) THEN
        ALTER TYPE jobstatus ADD VALUE 'completed';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'completed_partial'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
    ) THEN
        ALTER TYPE jobstatus ADD VALUE 'completed_partial';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'failed_retryable'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
    ) THEN
        ALTER TYPE jobstatus ADD VALUE 'failed_retryable';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'failed_terminal'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
    ) THEN
        ALTER TYPE jobstatus ADD VALUE 'failed_terminal';
    END IF;
END $$;

-- Add new columns for retries/claims (idempotent)
ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;
ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMP;
ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS last_error VARCHAR;
ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMP;
ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS worker_id VARCHAR;

-- Map old statuses to new ones so ORM can deserialize safely
UPDATE generationjob SET status = 'completed'::jobstatus WHERE status::text = 'succeeded';
UPDATE generationjob SET status = 'failed_terminal'::jobstatus WHERE status::text IN ('failed', 'cancelled');

-- Verify enum values
SELECT enumlabel FROM pg_enum
WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobstatus')
ORDER BY enumsortorder;

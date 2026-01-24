-- Migration: Add GENERATE_QUESTION_POOL and REFRESH_QUESTION_POOL to JobKind enum
-- Run this SQL script directly on your PostgreSQL database

-- Add GENERATE_QUESTION_POOL if it doesn't exist
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

-- Add REFRESH_QUESTION_POOL if it doesn't exist
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

-- Verify the enum values
SELECT enumlabel FROM pg_enum 
WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
ORDER BY enumsortorder;

-- Migration: Fix enum case mismatch - change to lowercase to match SQLAlchemy serialization
-- SQLAlchemy sends lowercase enum member names, so database should match

-- First, check current enum values
SELECT enumlabel FROM pg_enum 
WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
ORDER BY enumsortorder;

-- Note: PostgreSQL doesn't allow renaming enum values directly
-- We need to:
-- 1. Add lowercase versions
-- 2. Update existing data
-- 3. Remove uppercase versions (if safe)

-- Add lowercase versions if they don't exist
DO $$ 
BEGIN
    -- Add generate_question_pool (lowercase) if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'generate_question_pool' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
    ) THEN
        ALTER TYPE jobkind ADD VALUE 'generate_question_pool';
    END IF;
    
    -- Add refresh_question_pool (lowercase) if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'refresh_question_pool' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
    ) THEN
        ALTER TYPE jobkind ADD VALUE 'refresh_question_pool';
    END IF;
END $$;

-- Update any existing records that use uppercase values to lowercase
UPDATE generationjob 
SET kind = 'generate_question_pool'::jobkind 
WHERE kind::text = 'GENERATE_QUESTION_POOL';

UPDATE generationjob 
SET kind = 'refresh_question_pool'::jobkind 
WHERE kind::text = 'REFRESH_QUESTION_POOL';

-- Verify the enum values
SELECT enumlabel FROM pg_enum 
WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'jobkind')
ORDER BY enumsortorder;

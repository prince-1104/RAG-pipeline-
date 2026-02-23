-- Migration: add document_id to generationjob and generatedquestion
-- Run this SQL script directly on your PostgreSQL database

ALTER TABLE generationjob ADD COLUMN IF NOT EXISTS document_id VARCHAR;
CREATE INDEX IF NOT EXISTS ix_generationjob_document_id ON generationjob (document_id);

ALTER TABLE generatedquestion ADD COLUMN IF NOT EXISTS document_id VARCHAR;
CREATE INDEX IF NOT EXISTS ix_generatedquestion_document_id ON generatedquestion (document_id);

-- Optional FK for generatedquestion.document_id (safe to add if document table exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'generatedquestion_document_id_fkey'
    ) THEN
        ALTER TABLE generatedquestion
        ADD CONSTRAINT generatedquestion_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE SET NULL;
    END IF;
END $$;

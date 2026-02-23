-- Migration: add prewarm run tracking tables for promoted question export.

CREATE TABLE IF NOT EXISTS prewarmrun (
    id VARCHAR PRIMARY KEY,
    status VARCHAR NOT NULL DEFAULT 'running',
    syllabus VARCHAR NOT NULL DEFAULT 'CBSE',
    syllabus_version VARCHAR NOT NULL DEFAULT '2024',
    promoted_count INTEGER NOT NULL DEFAULT 0,
    pools_created INTEGER NOT NULL DEFAULT 0,
    metadata_json VARCHAR NULL,
    error_message VARCHAR NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP NULL
);

CREATE INDEX IF NOT EXISTS ix_prewarmrun_status ON prewarmrun(status);
CREATE INDEX IF NOT EXISTS ix_prewarmrun_syllabus ON prewarmrun(syllabus);
CREATE INDEX IF NOT EXISTS ix_prewarmrun_syllabus_version ON prewarmrun(syllabus_version);

CREATE TABLE IF NOT EXISTS prewarmrunquestion (
    id VARCHAR PRIMARY KEY,
    prewarm_id VARCHAR NOT NULL REFERENCES prewarmrun(id) ON DELETE CASCADE,
    question_id VARCHAR NOT NULL REFERENCES poolquestion(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_prewarmrunquestion_prewarm_id ON prewarmrunquestion(prewarm_id);
CREATE INDEX IF NOT EXISTS ix_prewarmrunquestion_question_id ON prewarmrunquestion(question_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_prewarmrunquestion_prewarm_question
    ON prewarmrunquestion(prewarm_id, question_id);

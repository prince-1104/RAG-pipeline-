-- Add canonical curriculum metadata fields for strict global RAG filtering.

ALTER TABLE document
    ADD COLUMN IF NOT EXISTS board TEXT DEFAULT 'CBSE',
    ADD COLUMN IF NOT EXISTS chapter_number INTEGER,
    ADD COLUMN IF NOT EXISTS chapter_name TEXT,
    ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en',
    ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'pdf',
    ADD COLUMN IF NOT EXISTS concepts_json TEXT,
    ADD COLUMN IF NOT EXISTS difficulty_hint TEXT;

ALTER TABLE chunk
    ADD COLUMN IF NOT EXISTS board TEXT DEFAULT 'CBSE',
    ADD COLUMN IF NOT EXISTS chapter_number INTEGER,
    ADD COLUMN IF NOT EXISTS chapter_name TEXT,
    ADD COLUMN IF NOT EXISTS syllabus_version TEXT,
    ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en',
    ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'pdf',
    ADD COLUMN IF NOT EXISTS concepts_json TEXT,
    ADD COLUMN IF NOT EXISTS difficulty_hint TEXT;

-- Backfill document.board from corpus.
UPDATE document d
SET board = c.board
FROM corpus c
WHERE d.corpus_id = c.id
  AND (d.board IS NULL OR d.board = '');

-- Backfill chunk metadata from document where available.
UPDATE chunk ch
SET
    board = COALESCE(ch.board, d.board, 'CBSE'),
    chapter_number = COALESCE(ch.chapter_number, d.chapter_number),
    chapter_name = COALESCE(ch.chapter_name, d.chapter_name, ch.chapter),
    syllabus_version = COALESCE(ch.syllabus_version, d.syllabus_version),
    language = COALESCE(ch.language, d.language, 'en'),
    source_type = COALESCE(ch.source_type, d.source_type, 'pdf'),
    difficulty_hint = COALESCE(ch.difficulty_hint, d.difficulty_hint)
FROM document d
WHERE ch.document_id = d.id;

CREATE INDEX IF NOT EXISTS ix_document_board ON document (board);
CREATE INDEX IF NOT EXISTS ix_document_chapter_number ON document (chapter_number);
CREATE INDEX IF NOT EXISTS ix_document_chapter_name ON document (chapter_name);
CREATE INDEX IF NOT EXISTS ix_document_language ON document (language);
CREATE INDEX IF NOT EXISTS ix_document_source_type ON document (source_type);
CREATE INDEX IF NOT EXISTS ix_document_difficulty_hint ON document (difficulty_hint);

CREATE INDEX IF NOT EXISTS ix_chunk_board ON chunk (board);
CREATE INDEX IF NOT EXISTS ix_chunk_chapter_number ON chunk (chapter_number);
CREATE INDEX IF NOT EXISTS ix_chunk_chapter_name ON chunk (chapter_name);
CREATE INDEX IF NOT EXISTS ix_chunk_syllabus_version ON chunk (syllabus_version);
CREATE INDEX IF NOT EXISTS ix_chunk_language ON chunk (language);
CREATE INDEX IF NOT EXISTS ix_chunk_source_type ON chunk (source_type);
CREATE INDEX IF NOT EXISTS ix_chunk_difficulty_hint ON chunk (difficulty_hint);

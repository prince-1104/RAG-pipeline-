-- Add canonical chapter identity fields to questionpool.
-- Canonical identity: (board, class_level, subject, syllabus_version, chapter_number)

ALTER TABLE questionpool
ADD COLUMN IF NOT EXISTS board VARCHAR;

ALTER TABLE questionpool
ADD COLUMN IF NOT EXISTS chapter_number INTEGER;

ALTER TABLE questionpool
ADD COLUMN IF NOT EXISTS chapter_name VARCHAR;

UPDATE questionpool
SET board = COALESCE(NULLIF(TRIM(board), ''), NULLIF(TRIM(syllabus), ''), 'CBSE')
WHERE board IS NULL OR TRIM(board) = '';

-- Best-effort backfill from chapter display text (e.g., "Chapter 5: Life Processes")
UPDATE questionpool
SET chapter_number = CAST(
    SUBSTRING(LOWER(chapter) FROM '(?:chapter|unit|ch)\s*([0-9]+)') AS INTEGER
)
WHERE chapter_number IS NULL
  AND chapter IS NOT NULL
  AND LOWER(chapter) ~ '(?:chapter|unit|ch)\s*[0-9]+';

UPDATE questionpool
SET chapter_name = NULLIF(
    TRIM(SUBSTRING(chapter FROM '^(?:[Cc]hapter|[Uu]nit|[Cc]h)\s*[0-9]+\s*[:\-]\s*(.+)$')),
    ''
)
WHERE chapter_name IS NULL
  AND chapter IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_questionpool_board ON questionpool(board);
CREATE INDEX IF NOT EXISTS ix_questionpool_chapter_number ON questionpool(chapter_number);
CREATE INDEX IF NOT EXISTS ix_questionpool_chapter_name ON questionpool(chapter_name);

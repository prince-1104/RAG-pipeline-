-- Rollback: remove serial_no and global sequence (use only if you previously ran add_question_serial_no)

DROP INDEX IF EXISTS ix_poolquestion_serial_no;
DROP INDEX IF EXISTS ix_generatedquestion_serial_no;
ALTER TABLE poolquestion DROP COLUMN IF EXISTS serial_no;
ALTER TABLE generatedquestion DROP COLUMN IF EXISTS serial_no;
DROP SEQUENCE IF EXISTS question_serial_seq;

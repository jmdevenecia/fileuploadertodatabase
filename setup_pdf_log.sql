-- ============================================================
-- setup_pdf_log.sql
-- Run this in Supabase SQL Editor before using upload_pdfs.py
-- ============================================================

CREATE TABLE IF NOT EXISTS pdf_ingestion_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    filepath    TEXT,
    checksum    TEXT UNIQUE,      -- MD5 hash to detect duplicates
    chunk_count INT,
    title       TEXT,
    category    TEXT,
    year        INT,
    authors     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- RLS: service role only
ALTER TABLE pdf_ingestion_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_all" ON pdf_ingestion_log
    FOR ALL TO service_role USING (true);

-- Helpful view: see all uploaded PDFs
CREATE OR REPLACE VIEW uploaded_pdfs AS
SELECT
    filename,
    title,
    category,
    year,
    authors,
    chunk_count,
    created_at
FROM pdf_ingestion_log
ORDER BY created_at DESC;

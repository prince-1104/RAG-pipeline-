from __future__ import annotations

import json
import gc

from sqlalchemy import delete, select
from sqlalchemy import func

from app.db.models import Chunk, Corpus, Document


def sanitize_text(text: str) -> str:
    """
    Remove characters that asyncpg/PostgreSQL rejects.
    psycopg2 was lenient about null bytes; asyncpg hard-rejects them.
    Critical for Jupyter-exported PDFs which contain \\x00 artifacts.
    """
    if not text:
        return text
    text = text.replace("\x00", "")
    text = "".join(
        ch for ch in text
        if ord(ch) >= 32 or ch in ("\n", "\r", "\t")
    )
    return text.strip()
from app.db.session import get_session_ctx
from app.indexing.pdf_parse import extract_pages, extract_pages_from_path
from app.indexing.qdrant_index import delete_document_points, upsert_document_chunks
from app.indexing.syllabus_chunking import syllabus_aware_chunk
from app.observability.logging import get_process_rss_mb, log
from app.settings import get_settings
from app.storage.object_store import get_object_store


async def index_document_qdrant(*, document_id: str) -> int:
    """
    Extract, chunk, persist Chunk rows, and upsert vectors into Qdrant.
    Returns the number of chunks created.
    """
    document_meta = await _load_document_meta(document_id)
    if not document_meta:
        return 0

    store = get_object_store()
    with store.open_local_pdf_path(document_meta["storage_uri"]) as pdf_path:
        pages = extract_pages_from_path(pdf_path)
    return await _index_document_qdrant_core(document_meta, pages)


async def index_document_qdrant_async(*, document_id: str) -> int:
    """
    Async variant for use in async endpoints.
    """
    document_meta = await _load_document_meta(document_id)
    if not document_meta:
        return 0

    store = get_object_store()
    pdf_bytes = await store.get_bytes(document_meta["storage_uri"])
    pages = extract_pages(pdf_bytes)
    return await _index_document_qdrant_core(document_meta, pages)


async def _load_document_meta(document_id: str) -> dict | None:
    async with get_session_ctx() as session:
        document = await session.get(Document, document_id)
        if not document:
            log.warning("document_not_found_for_indexing", document_id=document_id)
            return None

        return {
            "document_id": document.id,
            "corpus_id": document.corpus_id,
            "storage_uri": document.storage_uri,
            "toc_json": document.toc_json,
            "board": document.board,
            "class_level": document.class_level,
            "subject": document.subject,
            "chapter": document.chapter,
            "chapter_number": document.chapter_number,
            "chapter_name": document.chapter_name,
            "syllabus_version": document.syllabus_version,
            "language": document.language,
            "source_type": document.source_type,
            "difficulty_hint": document.difficulty_hint,
        }


async def _index_document_qdrant_core(document_meta: dict, pages: list) -> int:
    toc = json.loads(document_meta["toc_json"]) if document_meta["toc_json"] else None
    document_id = document_meta["document_id"]
    corpus_id = document_meta["corpus_id"]
    class_level = document_meta["class_level"]
    subject = document_meta["subject"]
    board = str(document_meta.get("board") or "CBSE")
    chapter_number = document_meta.get("chapter_number")
    chapter_name = document_meta.get("chapter_name")
    syllabus_version = document_meta.get("syllabus_version")
    language = str(document_meta.get("language") or "en")
    source_type = str(document_meta.get("source_type") or "pdf")
    difficulty_hint = document_meta.get("difficulty_hint")

    if not class_level or not subject:
        log.warning(
            "document_missing_tags",
            document_id=document_id,
            corpus_id=corpus_id,
        )
        return 0

    settings = get_settings()
    produced = syllabus_aware_chunk(
        pages=pages,
        class_level=int(class_level),
        subject=str(subject).strip(),
        toc=toc,
        max_tokens=int(settings.embedding_chunk_tokens),
        overlap_tokens=int(settings.embedding_chunk_overlap_tokens),
        hard_cap_tokens=int(settings.embedding_hard_cap_tokens),
    )
    if not produced:
        return 0

    rss_mb = get_process_rss_mb()
    if rss_mb is not None:
        log.info(
            "index_memory",
            stage="post_chunking",
            document_id=document_id,
            corpus_id=corpus_id,
            chunks=len(produced),
            rss_mb=round(rss_mb, 2),
        )

    override_chapter = document_meta.get("chapter")
    if override_chapter and str(override_chapter).strip():
        ch_name = str(override_chapter).strip()
        for ch in produced:
            ch.chapter = ch_name

    async with get_session_ctx() as session:
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.commit()

        for ch in produced:
            chunk_text = sanitize_text(ch.text)
            topics_clean = [sanitize_text(t) for t in ch.topics] if ch.topics else None
            chunk = Chunk(
                corpus_id=corpus_id,
                document_id=document_id,
                board=board,
                class_level=ch.class_level,
                subject=ch.subject,
                chapter=sanitize_text(ch.chapter) if ch.chapter else ch.chapter,
                chapter_number=chapter_number,
                chapter_name=sanitize_text(chapter_name or ch.chapter) if (chapter_name or ch.chapter) else None,
                syllabus_version=syllabus_version,
                language=language,
                source_type=source_type,
                concepts_json=json.dumps(topics_clean) if topics_clean else None,
                difficulty_hint=difficulty_hint,
                topics_json=json.dumps(topics_clean) if topics_clean else None,
                page_start=ch.page_start,
                page_end=ch.page_end,
                text=chunk_text,
                text_hash=ch.text_hash,
            )
            session.add(chunk)
        await session.commit()

        chunk_count = (await session.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        )).scalar_one() or 0
        corpus = await session.get(Corpus, corpus_id)
        doc_obj = await session.get(Document, document_id)

    if not corpus or not doc_obj:
        log.warning(
            "qdrant_upsert_skipped_missing_metadata",
            document_id=document_id,
            corpus_id=corpus_id,
        )
        return int(chunk_count or 0)

    delete_document_points(document_id=document_id)
    await upsert_document_chunks(document_id=document_id, corpus=corpus, document=doc_obj)
    gc.collect()

    rss_mb = get_process_rss_mb()
    if rss_mb is not None:
        log.info(
            "index_memory",
            stage="post_embedding",
            document_id=document_id,
            corpus_id=corpus_id,
            rss_mb=round(rss_mb, 2),
        )
    return int(chunk_count or 0)

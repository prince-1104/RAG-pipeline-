from __future__ import annotations

import json
import gc

from sqlmodel import delete, select
from sqlalchemy import func

from app.db.models import Chunk, Corpus, Document
from app.db.session import get_session
from app.indexing.pdf_parse import extract_pages, extract_pages_from_path
from app.indexing.qdrant_index import delete_document_points, upsert_document_chunks
from app.indexing.syllabus_chunking import syllabus_aware_chunk
from app.observability.logging import log
from app.settings import get_settings
from app.storage.object_store import get_object_store


def index_document_qdrant(*, document_id: str) -> int:
    """
    Extract, chunk, persist Chunk rows, and upsert vectors into Qdrant.
    Returns the number of chunks created.
    """
    document_meta = _load_document_meta(document_id)
    if not document_meta:
        return 0

    store = get_object_store()
    with store.open_local_pdf_path(document_meta["storage_uri"]) as pdf_path:
        pages = extract_pages_from_path(pdf_path)
    return _index_document_qdrant_core(document_meta, pages)


async def index_document_qdrant_async(*, document_id: str) -> int:
    """
    Async variant for use in async endpoints.
    """
    document_meta = _load_document_meta(document_id)
    if not document_meta:
        return 0

    store = get_object_store()
    pdf_bytes = await store.get_bytes(document_meta["storage_uri"])
    pages = extract_pages(pdf_bytes)
    return _index_document_qdrant_core(document_meta, pages)


def _load_document_meta(document_id: str) -> dict | None:
    with get_session() as session:
        document = session.get(Document, document_id)
        if not document:
            log.warning("document_not_found_for_indexing", document_id=document_id)
            return None

        return {
            "document_id": document.id,
            "corpus_id": document.corpus_id,
            "storage_uri": document.storage_uri,
            "toc_json": document.toc_json,
            "class_level": document.class_level,
            "subject": document.subject,
            "chapter": document.chapter,
        }


def _index_document_qdrant_core(document_meta: dict, pages: list) -> int:
    toc = json.loads(document_meta["toc_json"]) if document_meta["toc_json"] else None
    document_id = document_meta["document_id"]
    corpus_id = document_meta["corpus_id"]
    class_level = document_meta["class_level"]
    subject = document_meta["subject"]

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
        max_tokens=settings.embedding_max_tokens,
    )
    if not produced:
        return 0

    override_chapter = document_meta.get("chapter")
    if override_chapter and str(override_chapter).strip():
        ch_name = str(override_chapter).strip()
        for ch in produced:
            ch.chapter = ch_name

    with get_session() as session:
        # Idempotency: remove existing chunks for this document
        session.exec(delete(Chunk).where(Chunk.document_id == document_id))
        session.commit()

        for ch in produced:
            chunk = Chunk(
                corpus_id=corpus_id,
                document_id=document_id,
                class_level=ch.class_level,
                subject=ch.subject,
                chapter=ch.chapter,
                topics_json=json.dumps(ch.topics) if ch.topics else None,
                page_start=ch.page_start,
                page_end=ch.page_end,
                text=ch.text,
                text_hash=ch.text_hash,
            )
            session.add(chunk)
        session.commit()

        chunk_count = session.exec(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        ).one() or 0
        corpus = session.get(Corpus, corpus_id)
        doc_obj = session.get(Document, document_id)

    if not corpus or not doc_obj:
        log.warning(
            "qdrant_upsert_skipped_missing_metadata",
            document_id=document_id,
            corpus_id=corpus_id,
        )
        return int(chunk_count or 0)

    delete_document_points(document_id=document_id)
    upsert_document_chunks(document_id=document_id, corpus=corpus, document=doc_obj)
    gc.collect()
    return int(chunk_count or 0)

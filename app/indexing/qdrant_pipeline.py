from __future__ import annotations

import json

from sqlmodel import delete, select

from app.db.models import Chunk, Corpus, Document
from app.db.session import get_session
from app.indexing.pdf_parse import extract_pages
from app.indexing.qdrant_index import delete_document_points, upsert_chunks
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
    pdf_bytes = _load_pdf_bytes(store, document_meta["storage_uri"])
    return _index_document_qdrant_core(document_meta, pdf_bytes)


async def index_document_qdrant_async(*, document_id: str) -> int:
    """
    Async variant for use in async endpoints.
    """
    document_meta = _load_document_meta(document_id)
    if not document_meta:
        return 0

    store = get_object_store()
    pdf_bytes = await store.get_bytes(document_meta["storage_uri"])
    return _index_document_qdrant_core(document_meta, pdf_bytes)


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
        }


def _index_document_qdrant_core(document_meta: dict, pdf_bytes: bytes) -> int:
    pages = extract_pages(pdf_bytes)
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

        persisted_chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document_id)
        ).all()
        corpus = session.get(Corpus, corpus_id)
        doc_obj = session.get(Document, document_id)

    if not corpus or not doc_obj:
        log.warning(
            "qdrant_upsert_skipped_missing_metadata",
            document_id=document_id,
            corpus_id=corpus_id,
        )
        return len(persisted_chunks)

    delete_document_points(document_id=document_id)
    upsert_chunks(chunks=persisted_chunks, corpus=corpus, document=doc_obj)
    return len(persisted_chunks)


def _load_pdf_bytes(store, uri: str) -> bytes:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(store.get_bytes(uri))
    raise RuntimeError("asyncio.run() cannot be called from a running event loop")

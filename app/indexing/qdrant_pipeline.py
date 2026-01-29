from __future__ import annotations

import json

from sqlmodel import delete, select

from app.db.models import Chunk, Corpus, Document
from app.db.session import get_session
from app.indexing.pdf_parse import extract_pages
from app.indexing.qdrant_index import delete_document_points, upsert_chunks
from app.indexing.syllabus_chunking import syllabus_aware_chunk
from app.observability.logging import log
from app.storage.object_store import get_object_store


def index_document_qdrant(*, document: Document) -> int:
    """
    Extract, chunk, persist Chunk rows, and upsert vectors into Qdrant.
    Returns the number of chunks created.
    """
    store = get_object_store()
    pdf_bytes = _load_pdf_bytes(store, document.storage_uri)
    pages = extract_pages(pdf_bytes)
    toc = json.loads(document.toc_json) if document.toc_json else None

    if not document.class_level or not document.subject:
        log.warning(
            "document_missing_tags",
            document_id=document.id,
            corpus_id=document.corpus_id,
        )
        return 0

    produced = syllabus_aware_chunk(
        pages=pages,
        class_level=int(document.class_level),
        subject=str(document.subject).strip(),
        toc=toc,
    )
    if not produced:
        return 0

    with get_session() as session:
        # Idempotency: remove existing chunks for this document
        session.exec(delete(Chunk).where(Chunk.document_id == document.id))
        session.commit()

        for ch in produced:
            chunk = Chunk(
                corpus_id=document.corpus_id,
                document_id=document.id,
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
            select(Chunk).where(Chunk.document_id == document.id)
        ).all()
        corpus = session.get(Corpus, document.corpus_id)
        doc_obj = session.get(Document, document.id)

    if not corpus or not doc_obj:
        log.warning(
            "qdrant_upsert_skipped_missing_metadata",
            document_id=document.id,
            corpus_id=document.corpus_id,
        )
        return len(persisted_chunks)

    delete_document_points(document_id=document.id)
    upsert_chunks(chunks=persisted_chunks, corpus=corpus, document=doc_obj)
    return len(persisted_chunks)


def _load_pdf_bytes(store, uri: str) -> bytes:
    import asyncio

    return asyncio.run(store.get_bytes(uri))

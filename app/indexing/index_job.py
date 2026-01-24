from __future__ import annotations

import json
from datetime import datetime

from sqlmodel import select

from app.db.models import Chunk, Document, GenerationJob
from app.db.session import get_session
from app.indexing.embeddings import embed_texts
from app.indexing.faiss_index import IndexMetadata, save_index
from app.indexing.pdf_parse import extract_pages
from app.indexing.syllabus_chunking import syllabus_aware_chunk
from app.observability.logging import log
from app.storage.object_store import get_object_store


def _update_job_progress(job_id: str, progress_data: dict) -> None:
    """Update job progress by writing to result_json (for RUNNING status)."""
    with get_session() as session:
        job = session.get(GenerationJob, job_id)
        if job:
            job.result_json = json.dumps(progress_data)
            session.add(job)
            session.commit()


def run_index_job(job: GenerationJob) -> None:
    """
    Index job:
    - Read PDFs from object store
    - Syllabus-aware chunking -> Chunk rows
    - Build per-(corpus,class,subject) FAISS indices
    """
    assert job.corpus_id is not None
    req = json.loads(job.request_json)
    doc_ids = req.get("document_ids")

    with get_session() as session:
        q = select(Document).where(Document.corpus_id == job.corpus_id)
        if doc_ids:
            q = q.where(Document.id.in_(doc_ids))
        docs = session.exec(q).all()

    if not docs:
        log.info("index_job_no_documents", job_id=job.id, corpus_id=job.corpus_id)
        _update_job_progress(job.id, {
            "phase": "complete",
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "error": "no_documents_found",
        })
        return

    # Update progress: starting
    _update_job_progress(job.id, {
        "phase": "extracting",
        "documents_total": len(docs),
        "documents_processed": 0,
        "chunks_created": 0,
    })

    # 1) chunk + store in DB
    store = get_object_store()
    all_chunks_by_bucket: dict[tuple[int, str], list[Chunk]] = {}
    total_chunks = 0

    for doc_idx, doc in enumerate(docs):
        pdf_bytes = _load_pdf_bytes(store, doc.storage_uri)
        pages = extract_pages(pdf_bytes)
        toc = json.loads(doc.toc_json) if doc.toc_json else None
        # class_level/subject are mandatory for filtering; reject doc if missing
        if not doc.class_level or not doc.subject:
            log.warning("document_missing_tags", document_id=doc.id, corpus_id=doc.corpus_id)
            continue

        produced = syllabus_aware_chunk(
            pages=pages,
            class_level=int(doc.class_level),
            subject=str(doc.subject).strip(),
            toc=toc,
        )

        with get_session() as session:
            for ch in produced:
                chunk = Chunk(
                    corpus_id=doc.corpus_id,
                    document_id=doc.id,
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
                all_chunks_by_bucket.setdefault((ch.class_level, ch.subject), []).append(chunk)
            session.commit()

        total_chunks += len(produced)
        
        # Update progress after each document
        _update_job_progress(job.id, {
            "phase": "chunking",
            "documents_total": len(docs),
            "documents_processed": doc_idx + 1,
            "chunks_created": total_chunks,
        })

        log.info(
            "document_chunked",
            document_id=doc.id,
            corpus_id=doc.corpus_id,
            chunks=len(produced),
            class_level=doc.class_level,
            subject=doc.subject,
        )

    # Update progress: starting embedding
    _update_job_progress(job.id, {
        "phase": "embedding",
        "documents_total": len(docs),
        "documents_processed": len(docs),
        "chunks_created": total_chunks,
        "chunks_embedded": 0,
    })

    # 2) build indices
    total_indexed = 0
    bucket_count = len(all_chunks_by_bucket)
    
    for bucket_idx, ((class_level, subject), chunks) in enumerate(all_chunks_by_bucket.items()):
        # reload persisted chunks with ids
        with get_session() as session:
            persisted = session.exec(
                select(Chunk).where(
                    Chunk.corpus_id == job.corpus_id,
                    Chunk.class_level == class_level,
                    Chunk.subject == subject,
                )
            ).all()

        texts = [c.text for c in persisted]
        if not texts:
            continue
        
        # Update progress: embedding this bucket
        _update_job_progress(job.id, {
            "phase": "embedding",
            "documents_total": len(docs),
            "documents_processed": len(docs),
            "chunks_created": total_chunks,
            "chunks_embedded": total_indexed,
            "buckets_total": bucket_count,
            "buckets_processed": bucket_idx,
        })
        
        vectors = embed_texts(texts)
        meta = IndexMetadata(
            corpus_id=job.corpus_id,
            class_level=class_level,
            subject=subject,
            dim=int(vectors.shape[1]),
            chunk_ids=[c.id for c in persisted],
            created_at_iso=datetime.utcnow().isoformat(),
        )
        # save_index is async; worker is sync. For now, run via a tiny event loop.
        import asyncio

        asyncio.run(save_index(meta, vectors))
        total_indexed += len(texts)
        
        log.info(
            "faiss_index_saved",
            corpus_id=job.corpus_id,
            class_level=class_level,
            subject=subject,
            vectors=len(texts),
        )
    
    # Final progress update
    _update_job_progress(job.id, {
        "phase": "complete",
        "documents_total": len(docs),
        "documents_processed": len(docs),
        "chunks_created": total_chunks,
        "chunks_indexed": total_indexed,
        "chunks_embedded": total_indexed,
    })


def _load_pdf_bytes(store, uri: str) -> bytes:
    import asyncio

    return asyncio.run(store.get_bytes(uri))



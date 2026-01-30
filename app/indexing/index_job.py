from __future__ import annotations

import json

from sqlmodel import select

from app.db.models import Document, GenerationJob
from app.db.session import get_session
from app.indexing.qdrant_pipeline import index_document_qdrant
from app.observability.logging import log


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
    - Upsert chunk embeddings into Qdrant
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

    total_chunks = 0

    for doc_idx, doc in enumerate(docs):
        produced_count = index_document_qdrant(document_id=doc.id)
        total_chunks += produced_count
        
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
            chunks=produced_count,
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

    # Final progress update
    _update_job_progress(job.id, {
        "phase": "complete",
        "documents_total": len(docs),
        "documents_processed": len(docs),
        "chunks_created": total_chunks,
        "chunks_indexed": total_chunks,
        "chunks_embedded": total_chunks,
    })



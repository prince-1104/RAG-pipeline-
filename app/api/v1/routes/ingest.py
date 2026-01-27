from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from app.api.v1.schemas.ingest import PDFIngestResponse
from app.db.models import Corpus, Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session
from app.observability.logging import log
from app.settings import get_settings
from app.storage.object_store import get_object_store

router = APIRouter()


def _get_or_create_niyukti_corpus(tenant_id: str, board: str = "CBSE") -> str:
    """
    Get or create a default corpus for Niyukti tenant.
    Returns corpus_id.
    """
    with get_session() as session:
        # Try to find existing corpus for this tenant
        corpus = session.exec(
            select(Corpus).where(Corpus.tenant_id == tenant_id).limit(1)
        ).first()
        
        if corpus:
            return corpus.id
        
        # Create new corpus
        corpus = Corpus(
            name=f"Niyukti Default ({board})",
            board=board,
            tenant_id=tenant_id,
            description=f"Default corpus for Niyukti tenant {tenant_id}",
        )
        session.add(corpus)
        session.commit()
        session.refresh(corpus)
        log.info("niyukti_corpus_created", corpus_id=corpus.id, tenant_id=tenant_id, board=board)
        return corpus.id


@router.post("/pdf", response_model=PDFIngestResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    doc_id: str = Form(..., description="Document ID provided by caller"),
    board: str = Form(..., description="Board identifier (e.g., 'CBSE')"),
    class_level: int = Form(..., ge=1, le=12, description="Class level (1-12)"),
    subject: str = Form(..., description="Subject name"),
    tenant_id: str = Form(None, description="Tenant ID (defaults to 'niyukti-platform')"),
):
    """
    Ingest a PDF file into R2 storage and trigger indexing pipeline.
    
    - Accepts multipart PDF file
    - Accepts metadata (board, class_level, subject) and doc_id
    - Uploads PDF to Cloudflare R2 with key structure: board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
    - Creates Document record in database
    - Enqueues index job for automatic chunking, embedding, and indexing
    - Returns success/failure response with job IDs for tracking
    
    Workflow:
    1. Upload PDF to R2
    2. Create Document record
    3. Enqueue index job (chunking → embedding → FAISS index)
    """
    # Default tenant_id for Niyukti
    if not tenant_id:
        tenant_id = "niyukti-platform"
    
    log.info(
        "pdf_ingest_endpoint_called",
        doc_id=doc_id,
        board=board,
        class_level=class_level,
        subject=subject,
        tenant_id=tenant_id,
        filename=file.filename if file else None,
    )
    
    settings = get_settings()
    
    # Validate storage backend
    if settings.storage_backend != "r2":
        error_msg = f"pdf_ingestion_requires_r2_backend_current_backend_is_{settings.storage_backend}"
        log.error("storage_backend_mismatch", storage_backend=settings.storage_backend, doc_id=doc_id)
        raise HTTPException(status_code=400, detail=error_msg)

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file_must_be_pdf")

    # Validate content type if provided
    if file.content_type and file.content_type != "application/pdf":
        log.warning(
            "unexpected_content_type",
            content_type=file.content_type,
            filename=file.filename,
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        log.error("failed_to_read_uploaded_file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail="failed_to_read_file") from e

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="file_is_empty")

    # Validate PDF header
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="file_is_not_a_valid_pdf")

    # Create ingestion job to track status
    job = None
    storage_uri = None
    document_id = None
    index_job_id = None

    try:
        with get_session() as session:
            # Create job with RUNNING status (synchronous upload)
            now = datetime.utcnow()
            job = GenerationJob(
                kind=JobKind.ingest,
                status=JobStatus.running,  # Start as running since we're doing it synchronously
                tenant_id=tenant_id,
                corpus_id=None,  # Will be set after corpus creation
                started_at=now,
                request_json=json.dumps({
                    "doc_id": doc_id,
                    "board": board,
                    "class_level": class_level,
                    "subject": subject,
                    "filename": file.filename,
                }),
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id

        # Upload to R2
        try:
            store = get_object_store()
            storage_uri = await store.put_pdf_with_key(
                board=board,
                class_level=class_level,
                subject=subject,
                doc_id=doc_id,
                content=content,
            )

            # Get or create corpus for this tenant
            corpus_id = _get_or_create_niyukti_corpus(tenant_id, board)
            if not corpus_id:
                log.error("corpus_creation_failed", tenant_id=tenant_id, board=board)
                raise HTTPException(status_code=500, detail="corpus_creation_failed")
            
            # Create Document record
            with get_session() as session:
                document = Document(
                    corpus_id=corpus_id,
                    filename=f"{doc_id}.pdf",
                    storage_uri=storage_uri,
                    class_level=class_level,
                    subject=subject,
                    toc_json=None,  # TOC can be added later if available
                )
                session.add(document)
                session.commit()
                session.refresh(document)
                document_id = document.id
                if not document.corpus_id:
                    log.error("document_missing_corpus_id", document_id=document_id, tenant_id=tenant_id)
                    raise HTTPException(status_code=500, detail="document_missing_corpus_id")
                
                # Update ingest job with corpus_id
                job = session.get(GenerationJob, job_id)
                if job:
                    job.corpus_id = corpus_id
                    session.commit()
            
            # Create index job
            with get_session() as session:
                index_job = GenerationJob(
                    kind=JobKind.index,
                    status=JobStatus.queued,
                    tenant_id=tenant_id,
                    corpus_id=corpus_id,
                    request_json=json.dumps({
                        "document_ids": [document_id],
                    }),
                )
                session.add(index_job)
                session.commit()
                session.refresh(index_job)
                index_job_id = index_job.id
                if not index_job.corpus_id:
                    log.error("index_job_missing_corpus_id", index_job_id=index_job_id, tenant_id=tenant_id)
                    raise HTTPException(status_code=500, detail="index_job_missing_corpus_id")

            # Mark ingest job as succeeded
            with get_session() as session:
                job = session.get(GenerationJob, job_id)
                if job:
                    job.status = JobStatus.succeeded
                    job.finished_at = datetime.utcnow()
                    job.result_json = json.dumps({
                        "storage_uri": storage_uri,
                        "doc_id": doc_id,
                        "document_id": document_id,
                        "index_job_id": index_job_id,
                        "corpus_id": corpus_id,
                    })
                    session.commit()

            log.info(
                "pdf_ingested_successfully",
                doc_id=doc_id,
                storage_uri=storage_uri,
                job_id=job_id,
                document_id=document_id,
                index_job_id=index_job_id,
                board=board,
                class_level=class_level,
                subject=subject,
                tenant_id=tenant_id,
            )

            return PDFIngestResponse(
                success=True,
                doc_id=doc_id,
                corpus_id=corpus_id,
                storage_uri=storage_uri,
                job_id=job_id,
                index_job_id=index_job_id,
                document_id=document_id,
            )

        except ValueError as e:
            # PDF validation errors
            error_msg = str(e)
            log.error(
                "pdf_validation_failed",
                doc_id=doc_id,
                error=error_msg,
                job_id=job_id,
            )
            if job_id:
                with get_session() as session:
                    job = session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed
                        job.finished_at = datetime.utcnow()
                        job.error_code = "validation_error"
                        job.error_message = error_msg
                        session.commit()
            raise HTTPException(status_code=400, detail=error_msg) from e

        except HTTPException as e:
            error_msg = str(e.detail)
            log.error(
                "ingest_corpus_error",
                doc_id=doc_id,
                error=error_msg,
                job_id=job_id,
            )
            if job_id:
                with get_session() as session:
                    job = session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed
                        job.finished_at = datetime.utcnow()
                        job.error_code = "corpus_error"
                        job.error_message = error_msg
                        session.commit()
            raise

        except Exception as e:
            # R2 upload errors
            error_msg = f"r2_upload_failed: {str(e)}"
            log.error(
                "r2_upload_failed",
                doc_id=doc_id,
                error=str(e),
                job_id=job_id,
            )
            if job_id:
                with get_session() as session:
                    job = session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed
                        job.finished_at = datetime.utcnow()
                        job.error_code = "upload_error"
                        job.error_message = error_msg
                        session.commit()
            raise HTTPException(status_code=500, detail=error_msg) from e

    except SQLAlchemyError as e:
        log.error(
            "database_error_during_ingestion",
            doc_id=doc_id,
            error=str(e),
        )
        # Try to mark job as failed if it was created
        if job and job.id:
            try:
                with get_session() as session:
                    db_job = session.get(GenerationJob, job.id)
                    if db_job:
                        db_job.status = JobStatus.failed
                        db_job.finished_at = datetime.utcnow()
                        db_job.error_code = "database_error"
                        db_job.error_message = str(e)
                        session.commit()
            except Exception:
                pass  # Best effort to update job status
        raise HTTPException(status_code=500, detail="database_error") from e

    except Exception as e:
        # Unexpected errors
        error_msg = f"unexpected_error: {str(e)}"
        log.error(
            "unexpected_error_during_ingestion",
            doc_id=doc_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        if job and job.id:
            try:
                with get_session() as session:
                    db_job = session.get(GenerationJob, job.id)
                    if db_job:
                        db_job.status = JobStatus.failed
                        db_job.finished_at = datetime.utcnow()
                        db_job.error_code = "unexpected_error"
                        db_job.error_message = error_msg
                        session.commit()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=error_msg) from e

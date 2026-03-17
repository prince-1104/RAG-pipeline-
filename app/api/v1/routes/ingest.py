from __future__ import annotations

import json
from datetime import datetime
import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

from app.api.v1.schemas.ingest import PDFIngestResponse
from app.db.models import Corpus, Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session_ctx
from app.observability.logging import log
from app.settings import get_settings
from app.storage.object_store import get_object_store

router = APIRouter()

_INGEST_SEMAPHORE: asyncio.Semaphore | None = None
_INGEST_LIMIT: int | None = None


def _get_ingest_semaphore() -> asyncio.Semaphore:
    global _INGEST_SEMAPHORE, _INGEST_LIMIT
    settings = get_settings()
    limit = max(1, int(settings.ingest_max_concurrency))
    if _INGEST_SEMAPHORE is None or _INGEST_LIMIT != limit:
        _INGEST_SEMAPHORE = asyncio.Semaphore(limit)
        _INGEST_LIMIT = limit
    return _INGEST_SEMAPHORE


def _normalize_ingest_contract(
    *,
    board: str | None,
    class_level: int | None,
    subject: str | None,
    chapter: str | None,
    chapter_name: str | None,
    chapter_number: int | None,
    syllabus_version: str | None,
    language: str | None,
    source_type: str | None,
) -> dict:
    settings = get_settings()
    chapter_str = (chapter or "").strip() or None
    chapter_name_clean = (chapter_name or "").strip() or None
    if not chapter_str:
        if chapter_number is not None and chapter_name_clean:
            chapter_str = f"Chapter {int(chapter_number)}: {chapter_name_clean}"
        elif chapter_number is not None:
            chapter_str = f"Chapter {int(chapter_number)}"
        elif chapter_name_clean:
            chapter_str = chapter_name_clean
    return {
        "board": (str(board).strip() if board else None) or "CBSE",
        "class_level": int(class_level) if class_level is not None else 10,
        "subject": (str(subject).strip() if subject else None) or "General",
        "chapter": chapter_str,
        "chapter_name": chapter_name_clean,
        "chapter_number": int(chapter_number) if chapter_number is not None else None,
        "syllabus_version": (syllabus_version or "").strip() or settings.ingest_default_syllabus_version,
        "language": (language or "").strip().lower() or settings.ingest_default_language,
        "source_type": (source_type or "").strip().lower() or settings.ingest_default_source_type,
    }


def _validate_ingest_contract(tenant_id: str, doc_id: str, meta: dict) -> None:
    settings = get_settings()
    check_fields = [
        ("board", meta.get("board")),
        ("class_level", meta.get("class_level")),
        ("chapter_number", meta.get("chapter_number")),
        ("chapter_name", meta.get("chapter_name")),
    ]
    missing = [f for f, v in check_fields if not v]
    if missing:
        log.warning(
            "ingest_metadata_missing",
            tenant_id=tenant_id,
            doc_id=doc_id,
            missing=missing,
            strict_mode=bool(settings.ingest_enforce_strict_metadata),
        )
    if settings.ingest_enforce_strict_metadata:
        required = ("board", "class_level", "subject", "syllabus_version", "language", "source_type")
        missing_required = [k for k in required if not meta.get(k)]
        if missing_required:
            raise HTTPException(
                status_code=400,
                detail={"code": "missing_required_ingest_metadata", "missing_fields": missing_required},
            )


async def _get_or_create_niyukti_corpus(tenant_id: str, board: str = "CBSE") -> str:
    """
    Get or create a default corpus for Niyukti tenant.
    Returns corpus_id.
    """
    async with get_session_ctx() as session:
        corpus = (await session.execute(
            select(Corpus).where(Corpus.tenant_id == tenant_id).limit(1)
        )).scalars().first()
        
        if corpus:
            return corpus.id
        
        corpus = Corpus(
            name=f"Niyukti Default ({board})",
            board=board,
            tenant_id=tenant_id,
            description=f"Default corpus for Niyukti tenant {tenant_id}",
        )
        session.add(corpus)
        await session.commit()
        await session.refresh(corpus)
        log.info("niyukti_corpus_created", corpus_id=corpus.id, tenant_id=tenant_id, board=board)
        return corpus.id


@router.post("/pdf", response_model=PDFIngestResponse, status_code=202)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    doc_id: str = Form(..., description="Document ID provided by caller"),
    board: str | None = Form("CBSE", description="Board identifier (e.g., 'CBSE')"),
    class_level: int | None = Form(10, description="Class level (1-12); default 10 for school uploads"),
    subject: str | None = Form("General", description="Subject name"),
    chapter: str | None = Form(None, description="Chapter name (optional)"),
    chapter_name: str | None = Form(None, description="Chapter title/name (optional)"),
    chapter_number: int | None = Form(None, description="Chapter number (optional)"),
    syllabus_version: str | None = Form(None, description="Syllabus version"),
    language: str | None = Form(None, description="Language"),
    source_type: str | None = Form(None, description="Source type (e.g. pdf/scanned_pdf)"),
    tenant_id: str = Form(None, description="Tenant ID (defaults to 'niyukti-platform')"),
):
    """
    Ingest a PDF file into R2 storage and trigger indexing pipeline.
    
    - Accepts multipart PDF file
    - Accepts metadata (board, class_level, subject) and doc_id
    - Uploads PDF to Cloudflare R2 with key structure: board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
    - Creates Document record in database
    - Enqueues index job for automatic chunking, embedding, and indexing
    - Returns 202 with job IDs for tracking
    
    Workflow:
    1. Upload PDF to R2
    2. Create Document record
    3. Enqueue Qdrant indexing (chunking → embedding → Qdrant upsert)
    """
    if not tenant_id:
        tenant_id = "niyukti-platform"
    
    normalized_meta = _normalize_ingest_contract(
        board=board,
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        chapter_name=chapter_name,
        chapter_number=chapter_number,
        syllabus_version=syllabus_version,
        language=language,
        source_type=source_type,
    )
    _validate_ingest_contract(tenant_id, doc_id, normalized_meta)
    chapter_str = normalized_meta["chapter"]

    log.info(
        "pdf_ingest_endpoint_called",
        doc_id=doc_id,
        board=normalized_meta["board"],
        class_level=class_level,
        subject=normalized_meta["subject"],
        chapter=chapter_str,
        chapter_name=normalized_meta["chapter_name"],
        chapter_number=normalized_meta["chapter_number"],
        syllabus_version=normalized_meta["syllabus_version"],
        language=normalized_meta["language"],
        source_type=normalized_meta["source_type"],
        tenant_id=tenant_id,
        filename=file.filename if file else None,
    )
    
    settings = get_settings()
    
    if settings.storage_backend != "r2":
        error_msg = f"pdf_ingestion_requires_r2_backend_current_backend_is_{settings.storage_backend}"
        log.error("storage_backend_mismatch", storage_backend=settings.storage_backend, doc_id=doc_id)
        raise HTTPException(status_code=400, detail=error_msg)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file_must_be_pdf")

    if file.content_type and file.content_type != "application/pdf":
        log.warning(
            "unexpected_content_type",
            content_type=file.content_type,
            filename=file.filename,
        )

    try:
        header = await file.read(4)
        await file.seek(0)
    except Exception as e:
        log.error("failed_to_read_uploaded_file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail="failed_to_read_file") from e

    if not header:
        raise HTTPException(status_code=400, detail="file_is_empty")

    if not header.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="file_is_not_a_valid_pdf")

    job = None
    job_id = None
    storage_uri = None
    document_id = None
    index_job_id = None

    try:
        ingest_semaphore = _get_ingest_semaphore()
        await ingest_semaphore.acquire()
        async with get_session_ctx() as session:
            now = datetime.utcnow()
            job = GenerationJob(
                kind=JobKind.ingest,
                status=JobStatus.running,
                tenant_id=tenant_id,
                corpus_id=None,
                started_at=now,
                request_json=json.dumps({
                    "doc_id": doc_id,
                    "board": normalized_meta["board"],
                    "class_level": class_level,
                    "subject": normalized_meta["subject"],
                    "chapter": chapter_str,
                    "chapter_name": normalized_meta["chapter_name"],
                    "chapter_number": normalized_meta["chapter_number"],
                    "syllabus_version": normalized_meta["syllabus_version"],
                    "language": normalized_meta["language"],
                    "source_type": normalized_meta["source_type"],
                    "filename": file.filename,
                }),
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = job.id

        try:
            store = get_object_store()
            storage_uri = await store.put_pdf_with_key_stream(
                board=normalized_meta["board"],
                class_level=class_level,
                subject=normalized_meta["subject"],
                doc_id=doc_id,
                fileobj=file.file,
                content_type=file.content_type,
            )

            corpus_id = await _get_or_create_niyukti_corpus(tenant_id, normalized_meta["board"])
            if not corpus_id:
                log.error("corpus_creation_failed", tenant_id=tenant_id, board=normalized_meta["board"])
                raise HTTPException(status_code=500, detail="corpus_creation_failed")
            
            async with get_session_ctx() as session:
                document = Document(
                    corpus_id=corpus_id,
                    filename=f"{doc_id}.pdf",
                    storage_uri=storage_uri,
                    class_level=class_level,
                    subject=normalized_meta["subject"],
                    chapter=chapter_str,
                    chapter_number=normalized_meta["chapter_number"],
                    chapter_name=normalized_meta["chapter_name"],
                    board=normalized_meta["board"],
                    syllabus=normalized_meta["board"],
                    syllabus_version=normalized_meta["syllabus_version"],
                    language=normalized_meta["language"],
                    source_type=normalized_meta["source_type"],
                    toc_json=None,
                )
                session.add(document)
                await session.commit()
                await session.refresh(document)
                document_id = document.id
                if not document.corpus_id:
                    log.error("document_missing_corpus_id", document_id=document_id, tenant_id=tenant_id)
                    raise HTTPException(status_code=500, detail="document_missing_corpus_id")
                
                job = await session.get(GenerationJob, job_id)
                if job:
                    job.corpus_id = corpus_id
                    await session.commit()
            
            async with get_session_ctx() as session:
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
                await session.commit()
                await session.refresh(index_job)
                index_job_id = index_job.id
                if not index_job.corpus_id:
                    log.error("index_job_missing_corpus_id", index_job_id=index_job_id, tenant_id=tenant_id)
                    raise HTTPException(status_code=500, detail="index_job_missing_corpus_id")

            async with get_session_ctx() as session:
                job = await session.get(GenerationJob, job_id)
                if job:
                    job.status = JobStatus.completed
                    job.finished_at = datetime.utcnow()
                    job.result_json = json.dumps({
                        "storage_uri": storage_uri,
                        "doc_id": doc_id,
                        "document_id": document_id,
                        "index_job_id": index_job_id,
                        "corpus_id": corpus_id,
                    })
                    await session.commit()

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
            error_msg = str(e)
            log.error(
                "pdf_validation_failed",
                doc_id=doc_id,
                error=error_msg,
                job_id=job_id,
            )
            if job_id:
                async with get_session_ctx() as session:
                    job = await session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed_terminal
                        job.finished_at = datetime.utcnow()
                        job.error_code = "validation_error"
                        job.error_message = error_msg
                        await session.commit()
            raise HTTPException(status_code=400, detail=error_msg) from e

        except HTTPException as e:
            error_msg = str(e.detail)
            log.error(
                "ingest_error",
                doc_id=doc_id,
                error=error_msg,
                job_id=job_id,
            )
            if job_id:
                async with get_session_ctx() as session:
                    job = await session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed_terminal
                        job.finished_at = datetime.utcnow()
                        job.error_code = "ingest_error"
                        job.error_message = error_msg
                        await session.commit()
            raise

        except Exception as e:
            error_msg = f"r2_upload_failed: {str(e)}"
            log.error(
                "r2_upload_failed",
                doc_id=doc_id,
                error=str(e),
                job_id=job_id,
            )
            if job_id:
                async with get_session_ctx() as session:
                    job = await session.get(GenerationJob, job_id)
                    if job:
                        job.status = JobStatus.failed_terminal
                        job.finished_at = datetime.utcnow()
                        job.error_code = "upload_error"
                        job.error_message = error_msg
                        await session.commit()
            raise HTTPException(status_code=500, detail=error_msg) from e

        finally:
            ingest_semaphore.release()

    except SQLAlchemyError as e:
        log.error(
            "database_error_during_ingestion",
            doc_id=doc_id,
            error=str(e),
        )
        if job_id:
            try:
                async with get_session_ctx() as session:
                    db_job = await session.get(GenerationJob, job_id)
                    if db_job:
                        db_job.status = JobStatus.failed_terminal
                        db_job.finished_at = datetime.utcnow()
                        db_job.error_code = "database_error"
                        db_job.error_message = str(e)
                        await session.commit()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="database_error") from e

    except Exception as e:
        error_msg = f"unexpected_error: {str(e)}"
        log.error(
            "unexpected_error_during_ingestion",
            doc_id=doc_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        if job_id:
            try:
                async with get_session_ctx() as session:
                    db_job = await session.get(GenerationJob, job_id)
                    if db_job:
                        db_job.status = JobStatus.failed_terminal
                        db_job.finished_at = datetime.utcnow()
                        db_job.error_code = "unexpected_error"
                        db_job.error_message = error_msg
                        await session.commit()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=error_msg) from e

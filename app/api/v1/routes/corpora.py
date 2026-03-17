from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request

from app.api.v1.schemas.corpora import (
    CorpusCreateRequest,
    CorpusResponse,
    DocumentCreateRequest,
    DocumentRegisterRequest,
    DocumentResponse,
    IndexJobRequest,
    IndexJobResponse,
)
from app.db.models import Corpus, Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session_ctx
from app.observability.logging import log
from app.settings import get_settings
from app.storage.object_store import get_object_store


router = APIRouter()


@router.post("", response_model=CorpusResponse)
async def create_corpus(request: Request, req: CorpusCreateRequest):
    async with get_session_ctx() as session:
        corpus = Corpus(name=req.name, description=req.description, tenant_id=req.tenant_id)
        session.add(corpus)
        try:
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail="create_corpus_db_error") from e
        await session.refresh(corpus)
        return CorpusResponse(corpus_id=corpus.id, name=corpus.name, board=corpus.board, tenant_id=corpus.tenant_id)


@router.get("/{corpus_id}", response_model=CorpusResponse)
async def get_corpus(corpus_id: str):
    async with get_session_ctx() as session:
        corpus = await session.get(Corpus, corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        return CorpusResponse(corpus_id=corpus.id, name=corpus.name, board=corpus.board, tenant_id=corpus.tenant_id)


@router.post("/{corpus_id}/documents", response_model=DocumentResponse)
async def upload_document(
    corpus_id: str,
    file: UploadFile = File(...),
    meta: str = Form("{}"),
):
    """
    Upload a CBSE PDF to this corpus (for future use when S3 is enabled).
    Note: In local mode, use /documents/register instead.
    Note: Rate limiting removed from this endpoint due to decorator compatibility issues with UploadFile.
    """
    settings = get_settings()
    if settings.storage_backend == "local":
        raise HTTPException(
            status_code=400,
            detail="upload_not_supported_in_local_mode_use_register_endpoint",
        )

    try:
        meta_obj = DocumentCreateRequest.model_validate(json.loads(meta))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid_meta: {e}") from e

    async with get_session_ctx() as session:
        corpus = await session.get(Corpus, corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")

        store = get_object_store()
        storage_uri = await store.put_pdf(
            tenant_id=corpus.tenant_id,
            corpus_id=corpus_id,
            filename=meta_obj.filename,
            content=await file.read(),
        )

        doc = Document(
            corpus_id=corpus_id,
            filename=meta_obj.filename,
            storage_uri=storage_uri,
            syllabus=meta_obj.syllabus,
            class_level=meta_obj.class_level,
            subject=meta_obj.subject,
            syllabus_version=meta_obj.syllabus_version,
            chapter=meta_obj.chapter,
            toc_json=json.dumps(meta_obj.toc) if meta_obj.toc else None,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        log.info("document_uploaded", corpus_id=corpus_id, document_id=doc.id, storage_uri=storage_uri)
        return DocumentResponse(
            document_id=doc.id,
            corpus_id=doc.corpus_id,
            filename=doc.filename,
            storage_uri=doc.storage_uri,
            syllabus=doc.syllabus,
            class_level=doc.class_level,
            subject=doc.subject,
            syllabus_version=doc.syllabus_version,
            chapter=doc.chapter,
        )


@router.post("/{corpus_id}/documents/register", response_model=DocumentResponse)
async def register_document(request: Request, corpus_id: str, req: DocumentRegisterRequest):
    """
    Register an existing local PDF (no upload).
    PDF must exist under PDF_BASE_PATH/cbse/{class}/{subject}/{filename}
    """
    settings = get_settings()
    if settings.storage_backend != "local":
        raise HTTPException(
            status_code=400,
            detail="register_endpoint_only_for_local_mode_use_upload_endpoint",
        )

    async with get_session_ctx() as session:
        corpus = await session.get(Corpus, corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")

        store = get_object_store()
        storage_uri = store.build_local_pdf_uri(
            class_level=req.class_level,
            subject=req.subject,
            filename=req.filename,
        )

        try:
            pdf_bytes = await store.get_bytes(storage_uri)
            if not pdf_bytes.startswith(b"%PDF"):
                raise HTTPException(status_code=400, detail="file_is_not_a_valid_pdf")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"pdf_not_found_at_expected_path_check_pdf_base_path_and_structure",
            ) from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise

        doc = Document(
            corpus_id=corpus_id,
            filename=req.filename,
            storage_uri=storage_uri,
            syllabus=req.syllabus,
            class_level=req.class_level,
            subject=req.subject,
            syllabus_version=req.syllabus_version,
            chapter=req.chapter,
            toc_json=json.dumps(req.toc) if req.toc else None,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        log.info(
            "document_registered",
            corpus_id=corpus_id,
            document_id=doc.id,
            storage_uri=storage_uri,
            class_level=req.class_level,
            subject=req.subject,
        )
        return DocumentResponse(
            document_id=doc.id,
            corpus_id=doc.corpus_id,
            filename=doc.filename,
            storage_uri=doc.storage_uri,
            syllabus=doc.syllabus,
            class_level=doc.class_level,
            subject=doc.subject,
            syllabus_version=doc.syllabus_version,
            chapter=doc.chapter,
        )


@router.post("/{corpus_id}/index-jobs", response_model=IndexJobResponse)
async def create_index_job(request: Request, corpus_id: str, req: IndexJobRequest):
    async with get_session_ctx() as session:
        corpus = await session.get(Corpus, corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        if corpus.tenant_id != req.tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")

        if req.document_ids:
            docs = (await session.execute(select(Document).where(Document.corpus_id == corpus_id, Document.id.in_(req.document_ids)))).scalars().all()
            if len(docs) != len(req.document_ids):
                raise HTTPException(status_code=400, detail="some_documents_not_found")

        job = GenerationJob(
            kind=JobKind.index,
            status=JobStatus.queued,
            tenant_id=req.tenant_id,
            corpus_id=corpus_id,
            request_json=json.dumps({"document_ids": req.document_ids}),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return IndexJobResponse(job_id=job.id, status=job.status.value)

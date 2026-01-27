from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from app.api.v1.schemas.admin import (
    AdminBatchGenerateQuestionsRequest,
    AdminBatchGenerateQuestionsResponse,
    AdminGenerateQuestionsRequest,
    AdminGenerateQuestionsResponse,
    DocumentJobStatus,
)
from app.db.models import Corpus, Document, GenerationJob, JobKind, JobStatus
from app.db.session import get_session
from app.observability.logging import log


router = APIRouter()


@router.post("/generate-questions", response_model=AdminGenerateQuestionsResponse)
def admin_generate_questions(request: Request, req: AdminGenerateQuestionsRequest):
    """
    Admin endpoint to trigger question generation for uploaded PDFs.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    - PDF upload must NEVER trigger LLM
    - Institute APIs must NEVER enqueue generation jobs
    - Admin endpoints are asynchronous and return immediately
    
    The admin can specify:
    - Document ID (PDF that was uploaded)
    - Class range (from_class to to_class)
    - Subject
    - Difficulty level
    - Number of questions
    - Question types (mcq, short, long)
    - Optional chapters filter
    
    Questions will be generated asynchronously and stored in:
    - Database (PoolQuestion table)
    - Redis (for fast retrieval by institutes)
    """
    # Get tenant_id from header (admin should be authenticated)
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    # Validate document exists
    with get_session() as session:
        document = session.get(Document, req.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="document_not_found")
        
        # Get corpus
        corpus = session.get(Corpus, document.corpus_id)
        if not corpus:
            raise HTTPException(status_code=404, detail="corpus_not_found")
        
        # Validate tenant matches
        if corpus.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="tenant_mismatch")
        
        # Validate class range
        if req.from_class > req.to_class:
            raise HTTPException(
                status_code=400,
                detail="from_class_must_be_less_than_or_equal_to_to_class"
            )
        
        # Validate document has required metadata
        if not document.class_level or not document.subject:
            raise HTTPException(
                status_code=400,
                detail="document_missing_required_metadata_class_level_and_subject_required"
            )
        
        # Create admin generation job
        # Generate questions for each class in the range
        job_data = {
            "document_id": req.document_id,
            "corpus_id": document.corpus_id,
            "from_class": req.from_class,
            "to_class": req.to_class,
            "subject": req.subject,
            "difficulty": req.difficulty,
            "question_count": req.question_count,
            "question_types": req.question_types,
            "chapters": req.chapters,
            "syllabus": document.syllabus or "CBSE",
            "syllabus_version": document.syllabus_version or "2024",
        }
        
        job = GenerationJob(
            kind=JobKind.admin_generate_questions,
            status=JobStatus.queued,
            tenant_id=tenant_id,
            corpus_id=document.corpus_id,
            request_json=json.dumps(job_data),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        
        log.info(
            "admin_question_generation_job_created",
            job_id=job.id,
            document_id=req.document_id,
            from_class=req.from_class,
            to_class=req.to_class,
            subject=req.subject,
            difficulty=req.difficulty,
            question_count=req.question_count,
            tenant_id=tenant_id,
        )
        
        return AdminGenerateQuestionsResponse(
            job_id=job.id,
            status="queued",
            message=f"Question generation job queued for document {req.document_id}. Generating {req.question_count} {req.difficulty} questions for classes {req.from_class}-{req.to_class} in {req.subject}."
        )


@router.post("/generate-questions/batch", response_model=AdminBatchGenerateQuestionsResponse)
def admin_batch_generate_questions(request: Request, req: AdminBatchGenerateQuestionsRequest):
    """
    Batch admin endpoint to trigger question generation for multiple uploaded PDFs.
    
    This is the ONLY place where LLM calls are allowed for question generation.
    - PDF upload must NEVER trigger LLM
    - Institute APIs must NEVER enqueue generation jobs
    - Admin endpoints are asynchronous and return immediately
    
    Supports:
    - Multiple document IDs (batch processing)
    - "Any" difficulty (generates for all difficulty levels)
    - "Descriptive" question type (maps to "short" and "long")
    
    Returns per-document job status for tracking progress.
    """
    # Get tenant_id from header (admin should be authenticated)
    tenant_id = request.headers.get("X-Tenant-ID", "niyukti-platform")
    
    # Map "descriptive" to ["short", "long"]
    mapped_question_types = []
    for qt in req.question_types:
        if qt == "descriptive":
            mapped_question_types.extend(["short", "long"])
        else:
            mapped_question_types.append(qt)
    # Remove duplicates while preserving order
    mapped_question_types = list(dict.fromkeys(mapped_question_types))
    
    # Determine difficulty levels to generate
    if req.difficulty == "any":
        difficulty_levels = ["easy", "medium", "hard"]
    else:
        difficulty_levels = [req.difficulty]
    
    # Validate class range
    if req.from_class > req.to_class:
        raise HTTPException(
            status_code=400,
            detail="from_class_must_be_less_than_or_equal_to_to_class"
        )
    
    document_statuses: list[DocumentJobStatus] = []
    jobs_created = 0
    
    with get_session() as session:
        # Process each document
        for document_id in req.document_ids:
            try:
                # Validate document exists
                document = session.get(Document, document_id)
                if not document:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed",
                        message=f"Document {document_id} not found",
                        error="document_not_found"
                    ))
                    continue
                
                # Get corpus
                corpus = session.get(Corpus, document.corpus_id)
                if not corpus:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed",
                        message=f"Corpus not found for document {document_id}",
                        error="corpus_not_found"
                    ))
                    continue
                
                # Validate tenant matches
                if corpus.tenant_id != tenant_id:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed",
                        message=f"Tenant mismatch for document {document_id}",
                        error="tenant_mismatch"
                    ))
                    continue
                
                # Validate document has required metadata
                if not document.class_level or not document.subject:
                    document_statuses.append(DocumentJobStatus(
                        document_id=document_id,
                        job_id=None,
                        status="failed",
                        message=f"Document {document_id} missing required metadata (class_level and subject)",
                        error="missing_metadata"
                    ))
                    continue
                
                # Use document's subject if not provided, or validate it matches
                subject_to_use = req.subject
                if document.subject and document.subject != req.subject:
                    # Log warning but proceed with requested subject
                    log.warning(
                        "subject_mismatch",
                        document_id=document_id,
                        document_subject=document.subject,
                        requested_subject=req.subject,
                    )
                
                # Create jobs for each difficulty level
                job_ids = []
                for difficulty in difficulty_levels:
                    # Calculate questions per difficulty (distribute evenly)
                    questions_per_difficulty = max(1, req.question_count // len(difficulty_levels))
                    
                    job_data = {
                        "document_id": document_id,
                        "corpus_id": document.corpus_id,
                        "from_class": req.from_class,
                        "to_class": req.to_class,
                        "subject": subject_to_use,
                        "difficulty": difficulty,
                        "question_count": questions_per_difficulty,
                        "question_types": mapped_question_types,
                        "chapters": req.chapters,
                        "syllabus": document.syllabus or "CBSE",
                        "syllabus_version": document.syllabus_version or "2024",
                    }
                    
                    job = GenerationJob(
                        kind=JobKind.admin_generate_questions,
                        status=JobStatus.queued,
                        tenant_id=tenant_id,
                        corpus_id=document.corpus_id,
                        request_json=json.dumps(job_data),
                    )
                    session.add(job)
                    session.commit()
                    session.refresh(job)
                    job_ids.append(job.id)
                    jobs_created += 1
                    
                    log.info(
                        "admin_batch_question_generation_job_created",
                        job_id=job.id,
                        document_id=document_id,
                        from_class=req.from_class,
                        to_class=req.to_class,
                        subject=subject_to_use,
                        difficulty=difficulty,
                        question_count=questions_per_difficulty,
                        tenant_id=tenant_id,
                    )
                
                # Create status message
                if len(difficulty_levels) == 1:
                    message = f"Successfully queued generation of {req.question_count} {difficulty_levels[0]} questions for classes {req.from_class}-{req.to_class} in {subject_to_use}"
                else:
                    total_questions = req.question_count * len(difficulty_levels)
                    message = f"Successfully queued generation of {total_questions} questions ({req.question_count} per difficulty) for classes {req.from_class}-{req.to_class} in {subject_to_use}"
                
                document_statuses.append(DocumentJobStatus(
                    document_id=document_id,
                    job_id=job_ids[0] if job_ids else None,  # Return first job ID for tracking
                    status="queued",
                    message=message,
                ))
                
            except Exception as e:
                log.exception(
                    "admin_batch_generation_error",
                    document_id=document_id,
                    error=str(e),
                )
                document_statuses.append(DocumentJobStatus(
                    document_id=document_id,
                    job_id=None,
                    status="failed",
                    message=f"Error processing document {document_id}: {str(e)}",
                    error=str(e)[:200],  # Truncate error message
                ))
    
    return AdminBatchGenerateQuestionsResponse(
        total_documents=len(req.document_ids),
        jobs_created=jobs_created,
        documents=document_statuses,
        message=f"Batch processing complete. {jobs_created} job(s) created for {len([d for d in document_statuses if d.status == 'queued'])} document(s)."
    )

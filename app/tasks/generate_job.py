from __future__ import annotations

import json

from app.api.v1.schemas.generation import GenerateRequest, GenerateResponse
from app.db.models import GeneratedQuestion, QuestionCitation
from app.db.session import get_session
from app.generation.sync import generate_questions_sync
from app.observability.logging import log


def run_generate_job(job) -> dict:
    """
    Async generation job:
    - executes same core pipeline as sync
    - returns JSON serializable result (stored in GenerationJob.result_json)
    """
    req = GenerateRequest.model_validate(json.loads(job.request_json))
    resp: GenerateResponse = generate_questions_sync(req)

    # Persist generated questions + citations for auditability (institutes care).
    if resp.questions:
        with get_session() as session:
            for q in resp.questions:
                tags = q.tags or {}
                dq = GeneratedQuestion(
                    job_id=job.id,
                    corpus_id=req.corpus_id,
                    class_level=req.blueprint.constraints.class_level,
                    subject=req.blueprint.constraints.subject,
                    chapter=str(tags.get("chapter") or req.blueprint.constraints.chapters[0]),
                    difficulty=q.difficulty,
                    qtype=q.qtype,
                    language=q.language,
                    question_json=q.model_dump_json(),
                    dedupe_key=str(tags.get("dedupe_key") or ""),
                )
                session.add(dq)
                session.commit()
                session.refresh(dq)

                for cit in q.citations:
                    session.add(
                        QuestionCitation(
                            question_id=dq.id,
                            chunk_id=cit.chunk_id,
                            document_id=cit.document_id,
                            storage_uri=cit.storage_uri,
                            page_start=cit.page_start,
                            page_end=cit.page_end,
                            excerpt=cit.excerpt,
                        )
                    )
                session.commit()

        log.info("generation_persisted", job_id=job.id, questions=len(resp.questions))

    return resp.model_dump(mode="json")



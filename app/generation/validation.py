from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from app.generation.schema import QUESTION_SET_SCHEMA
from app.observability.logging import log


_validator = Draft202012Validator(QUESTION_SET_SCHEMA)


@dataclass
class ValidationIssue:
    code: str
    message: str
    details: dict[str, Any] | None = None


def validate_strict_schema(payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for err in sorted(_validator.iter_errors(payload), key=lambda e: e.path):
        issues.append(ValidationIssue(code="SCHEMA_INVALID", message=err.message, details={"path": list(err.path)}))
    return issues


def basic_difficulty_heuristic(stem: str) -> str:
    """
    Not perfect, but catches obvious mismatches:
    - 'define/what is/list' tends easy
    - 'derive/prove/evaluate with steps' tends hard
    """
    s = stem.lower()
    if re.search(r"\b(define|what is|list|state)\b", s):
        return "easy"
    if re.search(r"\b(derive|prove|evaluate|calculate|justify)\b", s):
        return "hard"
    return "medium"


def ensure_mandatory_citations(payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for i, q in enumerate(payload.get("questions", []) or []):
        citations = q.get("citations") or []
        if not citations:
            issues.append(ValidationIssue(code="MISSING_CITATIONS", message="Question missing citations", details={"i": i}))
    return issues


def parse_questions_from_llm_response(
    *,
    raw: str,
    difficulty: str,
    question_type: str,
    language: str,
) -> list:
    """
    Parse the LLM's raw text response into a list of SchoolQuestion objects.
    Strips markdown fences, parses JSON, and maps each dict to SchoolQuestion.
    """
    from app.api.v1.schemas.school import SchoolQuestion

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.error("school_llm_parse_failed", raw_response=raw[:500])
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="LLM returned invalid question format")

    # Handle both bare array and {"questions": [...]} wrapper
    if isinstance(parsed, dict) and "questions" in parsed:
        parsed = parsed["questions"]

    if not isinstance(parsed, list):
        log.error("school_llm_not_a_list", raw_response=raw[:500])
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="LLM returned invalid question format")

    questions: list[SchoolQuestion] = []
    for item in parsed:
        q_type = item.get("type", question_type if question_type != "mixed" else "mcq")
        questions.append(
            SchoolQuestion(
                id=str(uuid.uuid4()),
                question=item.get("question", ""),
                type=q_type,
                difficulty=item.get("difficulty", difficulty),
                options=item.get("options"),
                correct_answer=item.get("correct_answer", ""),
                explanation=item.get("explanation"),
                source_chapter=item.get("source_chapter"),
                language=language,
            )
        )
    return questions



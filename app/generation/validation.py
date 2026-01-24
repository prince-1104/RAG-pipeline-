from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from app.generation.schema import QUESTION_SET_SCHEMA


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



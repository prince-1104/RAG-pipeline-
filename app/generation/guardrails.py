from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
from sqlmodel import select

from app.db.models import GeneratedQuestion
from app.db.session import get_session
from app.indexing.qdrant_index import embed_texts
from app.observability.logging import log


def dedupe_key_for(stem: str, qtype: str, chapter: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]+", " ", stem.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    base = f"{qtype}::{chapter}::{norm}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


@dataclass
class GuardrailReject:
    reason: str
    details: dict


def reject_if_exact_duplicate(*, corpus_id: str, key: str) -> GuardrailReject | None:
    with get_session() as session:
        existing = session.exec(
            select(GeneratedQuestion).where(GeneratedQuestion.corpus_id == corpus_id, GeneratedQuestion.dedupe_key == key).limit(1)
        ).first()
    if existing:
        return GuardrailReject(reason="DUPLICATE_QUESTION", details={"dedupe_key": key, "existing_question_id": existing.id})
    return None


def reject_if_embedding_duplicate(
    *,
    corpus_id: str,
    stem: str,
    recent_limit: int = 200,
    similarity_threshold: float = 0.92,
) -> GuardrailReject | None:
    """
    Practical dedupe:
    - compare new stem embedding against recent stems in corpus
    - reject if cosine similarity exceeds threshold
    """
    with get_session() as session:
        recent = session.exec(
            select(GeneratedQuestion)
            .where(GeneratedQuestion.corpus_id == corpus_id)
            .order_by(GeneratedQuestion.created_at.desc())
            .limit(recent_limit)
        ).all()

    if not recent:
        return None

    # Pull stems from stored JSON (best-effort). This keeps schema evolution flexible.
    stems: list[str] = []
    for q in recent:
        try:
            import json

            stems.append(str(json.loads(q.question_json).get("stem") or ""))
        except Exception:  # noqa: BLE001
            continue
    stems = [s for s in stems if s]
    if not stems:
        return None

    # Fast exact-stem check to avoid embedding duplicates
    stem_norm = stem.strip().lower()
    for s in stems:
        if stem_norm and s.strip().lower() == stem_norm:
            return GuardrailReject(
                reason="DUPLICATE_QUESTION",
                details={"match_type": "exact_stem"},
            )

    # Deduplicate stems list before embedding to avoid redundant vectors
    seen = set()
    unique_stems: list[str] = []
    for s in stems:
        key = s.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_stems.append(s)
    stems = unique_stems

    try:
        # qdrant_index.embed_texts returns (idx, vector) tuples; unwrap to vectors
        pairs = embed_texts([stem] + stems)
        pairs = sorted(pairs, key=lambda x: x[0])
        vecs = [v for _, v in pairs]
        embs = np.asarray(vecs, dtype=np.float32)
    except Exception as exc:
        log.warning("guardrail_embedding_failed", error=str(exc))
        return None
    qv = embs[0]
    mat = embs[1:]
    sims: np.ndarray = mat @ qv
    best = float(np.max(sims))
    if best >= similarity_threshold:
        return GuardrailReject(reason="DUPLICATE_QUESTION", details={"similarity": best, "threshold": similarity_threshold})
    return None


def reject_if_low_support(*, answer: str | None, explanation: str | None, excerpt: str) -> GuardrailReject | None:
    """
    Basic factual support check:
    - require overlap of key tokens between (answer+explanation) and excerpt
    - reject obvious “free-floating” answers
    """
    text = " ".join([answer or "", explanation or ""]).strip().lower()
    if not text:
        return None
    ex = excerpt.lower()

    def toks(s: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", s) if len(t) >= 5}

    a = toks(text)
    b = toks(ex)
    if not a or not b:
        return None
    overlap = len(a & b) / max(1, len(a))
    if overlap < 0.18:
        return GuardrailReject(reason="INSUFFICIENT_CONTEXT", details={"token_overlap": overlap})
    return None



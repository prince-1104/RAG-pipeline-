from __future__ import annotations

import math
import re
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlmodel import select

from app.db.models import Chunk, Corpus
from app.db.session import get_session
from app.indexing.qdrant_index import embed_texts, get_qdrant_client
from app.observability.logging import log
from app.settings import get_settings


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    confidence: float


def retrieve(
    *,
    corpus_id: str,
    class_level: int,
    subject: str,
    chapters: list[str],
    query: str,
    top_k: int,
) -> RetrievalResult:
    """
    Mandatory retrieval:
    - metadata filter FIRST (class/subject/chapters)
    - then similarity search on candidate subset
    """
    settings = get_settings()

    # 1) Qdrant retrieval with strict filters (corpus, class, subject, chapter)
    # When chapters == ["Unknown Chapter"] skip chapter filter.
    _skip_chapter_filter = (
        len(chapters) == 1 and _normalize_chapter_name(chapters[0]) == "unknown chapter"
    )
    normalized_subject = subject.strip()
    with get_session() as session:
        corpus = session.get(Corpus, corpus_id)
    if not corpus:
        return RetrievalResult(query=query, chunks=[], confidence=0.0)
    normalized_requested = [_normalize_chapter_name(ch) for ch in chapters]

    must: list[FieldCondition] = [
        FieldCondition(key="corpus_id", match=MatchValue(value=corpus_id)),
        FieldCondition(key="tenant_id", match=MatchValue(value=corpus.tenant_id)),
        FieldCondition(key="class_level", match=MatchValue(value=class_level)),
        FieldCondition(key="subject", match=MatchValue(value=normalized_subject)),
    ]
    should = (
        [FieldCondition(key="chapter", match=MatchValue(value=ch)) for ch in chapters]
        if not _skip_chapter_filter
        else None
    )
    flt = Filter(must=must, should=should)

    client = get_qdrant_client()
    vector = embed_texts([query])[0]
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        query_filter=flt,
        limit=max(1, min(top_k, 100)),
    )

    # Fallback: if no hits and chapter filter was applied, re-run without chapters
    if not hits and not _skip_chapter_filter:
        flt = Filter(must=must)
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            query_filter=flt,
            limit=max(1, min(top_k * 2, 200)),
        )

    # Filter hits by flexible chapter matching if needed
    filtered_hits = []
    for h in hits:
        payload = h.payload or {}
        stored_chapter = str(payload.get("chapter") or "").strip()
        if _skip_chapter_filter:
            filtered_hits.append(h)
            continue
        normalized_stored = _normalize_chapter_name(stored_chapter)
        for req_ch in normalized_requested:
            if req_ch in normalized_stored or normalized_stored in req_ch:
                filtered_hits.append(h)
                break
            req_num = _extract_chapter_number(req_ch)
            stored_num = _extract_chapter_number(normalized_stored)
            if req_num and stored_num and req_num == stored_num:
                filtered_hits.append(h)
                break

    if not filtered_hits:
        log.warning(
            "retrieval_no_chapter_match",
            corpus_id=corpus_id,
            class_level=class_level,
            subject=subject,
            requested_chapters=chapters,
        )
        return RetrievalResult(query=query, chunks=[], confidence=0.0)

    chunk_ids = []
    for h in filtered_hits:
        payload = h.payload or {}
        cid = payload.get("chunk_id")
        if cid:
            chunk_ids.append(str(cid))

    if not chunk_ids:
        return RetrievalResult(query=query, chunks=[], confidence=0.0)

    with get_session() as session:
        chunks = session.exec(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()
    chunk_by_id = {str(c.id): c for c in chunks}

    results: list[RetrievedChunk] = []
    for h in filtered_hits:
        payload = h.payload or {}
        cid = str(payload.get("chunk_id") or "")
        ch = chunk_by_id.get(cid)
        if not ch:
            continue
        results.append(RetrievedChunk(chunk=ch, score=float(h.score)))

    conf = float(results[0].score) if results else 0.0
    conf = max(0.0, min(1.0, (conf + 1.0) / 2.0)) if not math.isnan(conf) else 0.0

    if conf < settings.min_retrieval_confidence:
        log.warning(
            "retrieval_low_confidence",
            corpus_id=corpus_id,
            class_level=class_level,
            subject=subject,
            confidence=conf,
        )

    return RetrievalResult(query=query, chunks=results, confidence=conf)


def _normalize_chapter_name(chapter: str) -> str:
    """Normalize chapter name for flexible matching."""
    if not chapter:
        return ""
    # Lowercase, strip whitespace, remove extra spaces
    normalized = re.sub(r"\s+", " ", chapter.strip().lower())
    return normalized


def _extract_chapter_number(chapter: str) -> str | None:
    """Extract chapter number from chapter name (e.g., 'Chapter 1' -> '1')."""
    if not chapter:
        return None
    # Match patterns like "chapter 1", "chapter 1:", "unit 1", etc.
    match = re.search(r"(?:chapter|unit|ch)\s*(\d+)", chapter.lower())
    return match.group(1) if match else None



from __future__ import annotations

import math
import re
from threading import Lock
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlmodel import select

from app.db.models import Chunk, Corpus
from app.db.session import get_session
from app.indexing.qdrant_index import embed_texts, get_qdrant_client
from app.observability.logging import log
from app.settings import get_settings

_QUERY_EMBED_CACHE_LOCK = Lock()
_QUERY_EMBED_CACHE: dict[tuple[str, str, str, str], list[float]] = {}
_QUERY_EMBED_CACHE_MAX = 512


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
    document_id: str | None = None,
    class_level: int,
    subject: str,
    chapters: list[str],
    query: str,
    top_k: int,
    board: str | None = None,
    syllabus_version: str | None = None,
    chapter_number: int | None = None,
) -> RetrievalResult:
    """
    Mandatory retrieval:
    - metadata filter FIRST (class/subject/chapters)
    - then similarity search on candidate subset
    """
    settings = get_settings()

    # 1) Qdrant retrieval with strict filters (metadata first, then vector search).
    # When chapters == ["Unknown Chapter"] skip chapter filter.
    _skip_chapter_filter = (
        len(chapters) == 1 and _normalize_chapter_name(chapters[0]) == "unknown chapter"
    )
    normalized_subject = _normalize_subject(subject)
    with get_session() as session:
        corpus = session.get(Corpus, corpus_id)
    if not corpus:
        return RetrievalResult(query=query, chunks=[], confidence=0.0)
    must: list[FieldCondition] = [
        FieldCondition(key="corpus_id", match=MatchValue(value=corpus_id)),
        FieldCondition(key="class_level", match=MatchValue(value=class_level)),
        FieldCondition(key="subject", match=MatchValue(value=normalized_subject)),
    ]
    if document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    if not settings.global_retrieval_ignore_tenant:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=corpus.tenant_id)))
    requested_board = (board or "").strip() or str(corpus.board or "").strip()
    if requested_board:
        must.append(FieldCondition(key="board", match=MatchValue(value=requested_board)))
    if syllabus_version:
        must.append(FieldCondition(key="syllabus_version", match=MatchValue(value=syllabus_version)))
    if chapter_number is not None:
        must.append(FieldCondition(key="chapter_number", match=MatchValue(value=int(chapter_number))))
        should = None
    elif _skip_chapter_filter:
        should = None
    else:
        should = []
        for ch in chapters:
            ch_str = str(ch).strip()
            if ch_str.isdigit():
                should.append(
                    FieldCondition(key="chapter_number", match=MatchValue(value=int(ch_str)))
                )
            else:
                should.append(FieldCondition(key="chapter", match=MatchValue(value=ch_str)))
    flt = Filter(must=must, should=should)

    client = get_qdrant_client()
    cache_key = (
        settings.qdrant_collection,
        settings.embedding_provider,
        settings.embedding_model,
        query,
    )
    with _QUERY_EMBED_CACHE_LOCK:
        cached = _QUERY_EMBED_CACHE.get(cache_key)
    if cached is None:
        vector = embed_texts([query])[0][1]
        with _QUERY_EMBED_CACHE_LOCK:
            _QUERY_EMBED_CACHE[cache_key] = vector
            if len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_MAX:
                oldest = next(iter(_QUERY_EMBED_CACHE))
                _QUERY_EMBED_CACHE.pop(oldest, None)
    else:
        vector = cached

    def _search(collection: str, flt_obj: Filter, limit: int) -> list:
        if hasattr(client, "search"):
            return client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=flt_obj,
                limit=limit,
            )
        if hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=flt_obj,
                limit=limit,
            )
            if hasattr(resp, "points"):
                return list(resp.points or [])
            if hasattr(resp, "result"):
                return list(resp.result or [])
            return list(resp or [])
        raise AttributeError("QdrantClient has no search or query_points")

    hits = _search(
        settings.qdrant_collection,
        flt,
        max(1, min(top_k, 100)),
    )
    filtered_hits = list(hits or [])
    if not filtered_hits:
        log.warning(
            "retrieval_no_in_scope_chunks",
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


def _normalize_subject(subject: str) -> str:
    """
    Normalize subject names to match stored chunk payloads.
    Mirrors Niyukti subject mapping and enforces consistent casing.
    """
    if not subject:
        return ""
    subject_clean = subject.strip()
    subject_lower = subject_clean.lower()
    mapping = {
        "maths": "Mathematics",
        "math": "Mathematics",
        "mathematics": "Mathematics",
        "science": "Science",
        "english": "English",
        "hindi": "Hindi",
        "social_science": "Social Science",
        "socialscience": "Social Science",
        "social science": "Social Science",
        "sst": "Social Science",
    }
    if subject_lower in mapping:
        return mapping[subject_lower]
    # Default: title case for multi-word subjects (e.g., "computer science")
    return " ".join([w.capitalize() for w in subject_clean.split()])


def _extract_chapter_number(chapter: str) -> str | None:
    """Extract chapter number from chapter name (e.g., 'Chapter 1' -> '1')."""
    if not chapter:
        return None
    if chapter.isdigit():
        return chapter
    # Match patterns like "chapter 1", "chapter 1:", "unit 1", etc.
    match = re.search(r"(?:chapter|unit|ch)\s*(\d+)", chapter.lower())
    return match.group(1) if match else None



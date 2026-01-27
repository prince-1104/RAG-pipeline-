from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass

import numpy as np
from sqlmodel import select

from app.db.models import Chunk
from app.db.session import get_session
from app.indexing.embeddings import embed_texts
from app.indexing.faiss_index import load_index
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
    try:
        # Handle async load_index in sync context
        # Since this is a sync function, we can safely use asyncio.run()
        # If there's already a running loop (shouldn't happen), we'll handle it
        try:
            meta, vectors, _index = asyncio.run(load_index(corpus_id, class_level, subject))
        except RuntimeError:
            # If asyncio.run() fails (running loop exists), create new loop in thread
            import threading
            import queue
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(load_index(corpus_id, class_level, subject))
                        result_queue.put(result)
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
            thread.join(timeout=30)
            
            if not exception_queue.empty():
                raise exception_queue.get()
            if not result_queue.empty():
                meta, vectors, _index = result_queue.get()
            else:
                raise RuntimeError("load_index timed out")
    except Exception as e:  # noqa: BLE001
        error_msg = str(e)[:500]
        log.warning("missing_index", corpus_id=corpus_id, class_level=class_level, subject=subject, error=error_msg, error_type=type(e).__name__)
        return RetrievalResult(query=query, chunks=[], confidence=0.0)
    if not meta.chunk_ids:
        return RetrievalResult(query=query, chunks=[], confidence=0.0)

    # 1) metadata filter (DB) -> candidate chunk_ids
    # Use flexible chapter matching to handle name variations.
    # When chapters == ["Unknown Chapter"] (no chapter data from caller), skip chapter
    # filter so we still retrieve from all (corpus, class, subject) content.
    _skip_chapter_filter = (
        len(chapters) == 1 and _normalize_chapter_name(chapters[0]) == "unknown chapter"
    )
    with get_session() as session:
        # Normalize requested chapters for matching
        normalized_requested = [_normalize_chapter_name(ch) for ch in chapters]
        normalized_subject = subject.strip()

        if _skip_chapter_filter:
            # No chapter info: filter only by corpus, class, subject
            all_chunks = session.exec(
                select(Chunk).where(
                    Chunk.corpus_id == corpus_id,
                    Chunk.class_level == class_level,
                )
            ).all()
            normalized_subject_lower = normalized_subject.lower().strip()
            candidates = [
                c for c in all_chunks
                if c.subject and c.subject.strip().lower() == normalized_subject_lower
            ]
            if candidates:
                log.info(
                    "retrieval_skip_chapter_filter",
                    corpus_id=corpus_id,
                    class_level=class_level,
                    subject=subject,
                    candidate_count=len(candidates),
                )
        else:
            # First try exact match (both subject and chapters)
            candidates = session.exec(
                select(Chunk).where(
                    Chunk.corpus_id == corpus_id,
                    Chunk.class_level == class_level,
                    Chunk.subject == normalized_subject,
                    Chunk.chapter.in_(chapters),
                )
            ).all()

        # If no exact matches (and we didn't skip chapter filter), try flexible matching
        if not candidates and not _skip_chapter_filter:
            # Try case-insensitive subject matching first
            all_chunks = session.exec(
                select(Chunk).where(
                    Chunk.corpus_id == corpus_id,
                    Chunk.class_level == class_level,
                )
            ).all()
            
            # Filter by normalized subject (case-insensitive, whitespace-normalized)
            normalized_subject_lower = normalized_subject.lower().strip()
            subject_matched_chunks = [
                c for c in all_chunks
                if c.subject and c.subject.strip().lower() == normalized_subject_lower
            ]
            
            if not subject_matched_chunks:
                # Log available subjects for debugging
                available_subjects = {c.subject for c in all_chunks if c.subject}
                log.warning(
                    "retrieval_subject_not_found",
                    corpus_id=corpus_id,
                    class_level=class_level,
                    requested_subject=subject,
                    available_subjects=list(available_subjects),
                )
                return RetrievalResult(query=query, chunks=[], confidence=0.0)
            
            all_chunks = subject_matched_chunks
            
            # Filter chunks where normalized chapter name matches any requested chapter
            candidates = []
            for chunk in all_chunks:
                normalized_stored = _normalize_chapter_name(chunk.chapter)
                # Check if any requested chapter matches (contains or is contained)
                for req_ch in normalized_requested:
                    if req_ch in normalized_stored or normalized_stored in req_ch:
                        candidates.append(chunk)
                        break
                    # Also check if chapter numbers match (e.g., "Chapter 1" matches "Chapter 1: Title")
                    req_num = _extract_chapter_number(req_ch)
                    stored_num = _extract_chapter_number(normalized_stored)
                    if req_num and stored_num and req_num == stored_num:
                        candidates.append(chunk)
                        break
            
            if candidates:
                log.info(
                    "retrieval_flexible_chapter_match",
                    corpus_id=corpus_id,
                    class_level=class_level,
                    subject=subject,
                    requested_chapters=chapters,
                    matched_count=len(candidates),
                )

    if not candidates:
        # Log available chapters for debugging (use normalized subject for query)
        with get_session() as session:
            # Get all chunks for this corpus/class, then filter by normalized subject
            all_chunks_for_logging = session.exec(
                select(Chunk)
                .where(
                    Chunk.corpus_id == corpus_id,
                    Chunk.class_level == class_level,
                )
            ).all()
            # Filter by normalized subject
            normalized_subject_lower = normalized_subject.lower().strip()
            matching_chunks = [
                c for c in all_chunks_for_logging
                if c.subject and c.subject.strip().lower() == normalized_subject_lower
            ]
            available_chapters = list({c.chapter for c in matching_chunks})[:20]
            available_subjects = list({c.subject for c in all_chunks_for_logging if c.subject})[:10]
            log.warning(
                "retrieval_no_chapter_match",
                corpus_id=corpus_id,
                class_level=class_level,
                subject=subject,
                requested_chapters=chapters,
                available_chapters=available_chapters,
            )
        return RetrievalResult(query=query, chunks=[], confidence=0.0)

    # map to vector rows
    row_by_chunk_id = {cid: i for i, cid in enumerate(meta.chunk_ids)}
    cand_rows: list[int] = [row_by_chunk_id[c.id] for c in candidates if c.id in row_by_chunk_id]
    if not cand_rows:
        # Some candidates exist in DB but not in index - index may be out of sync
        missing_in_index = [c.id for c in candidates if c.id not in row_by_chunk_id]
        log.warning(
            "retrieval_chunks_missing_in_index",
            corpus_id=corpus_id,
            class_level=class_level,
            subject=subject,
            missing_count=len(missing_in_index),
            total_candidates=len(candidates),
            suggestion="Re-run indexing job to sync index with database",
        )
        return RetrievalResult(query=query, chunks=[], confidence=0.0)

    # 2) similarity search restricted to filtered rows
    qv = embed_texts([query])[0]  # normalized
    cand_vecs = vectors[np.array(cand_rows, dtype=np.int64)]
    scores = cand_vecs @ qv
    top_idx = np.argsort(-scores)[: max(1, min(top_k, scores.shape[0]))]

    results: list[RetrievedChunk] = []
    for idx in top_idx:
        row = cand_rows[int(idx)]
        chunk_id = meta.chunk_ids[row]
        ch = next((c for c in candidates if c.id == chunk_id), None)
        if ch is None:
            continue
        results.append(RetrievedChunk(chunk=ch, score=float(scores[int(idx)])))

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



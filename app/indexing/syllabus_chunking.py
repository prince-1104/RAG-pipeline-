from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.indexing.pdf_parse import PageText


_CHAPTER_RE = re.compile(r"^\s*(chapter|unit)\s+(\d+)\s*[:.\-]?\s*(.+)$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 \-:,()]{6,}$")


@dataclass
class ChunkOut:
    class_level: int
    subject: str
    chapter: str
    page_start: int
    page_end: int
    text: str
    text_hash: str
    topics: list[str]


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _detect_chapter_from_page(page_text: str) -> str | None:
    """
    Best-effort chapter detection from visible text.
    Real production uses TOC + curated CBSE structure; this is a safe baseline.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for ln in lines[:12]:
        m = _CHAPTER_RE.match(ln)
        if m:
            title = _normalize_ws(m.group(3))
            return f"Chapter {m.group(2)}: {title}" if title else f"Chapter {m.group(2)}"
    return None


def syllabus_aware_chunk(
    *,
    pages: list[PageText],
    class_level: int,
    subject: str,
    toc: list[dict] | None,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[ChunkOut]:
    """
    Syllabus-aware chunking:
    - Prefer TOC ranges for chapter boundaries when provided
    - Fall back to chapter heading detection heuristics
    - Chunk *within* a chapter using paragraph-ish boundaries, not fixed tokens
    """
    if toc:
        return _chunk_by_toc(
            pages=pages,
            class_level=class_level,
            subject=subject,
            toc=toc,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
    return _chunk_by_heading(
        pages=pages,
        class_level=class_level,
        subject=subject,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return parts


def _topics_from_text(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    topics: list[str] = []
    for ln in lines[:20]:
        if _HEADING_RE.match(ln) and len(ln) <= 80:
            topics.append(_normalize_ws(ln.title()))
    return topics[:6]


def _chunk_by_toc(
    *,
    pages: list[PageText],
    class_level: int,
    subject: str,
    toc: list[dict],
    max_chars: int,
    overlap_chars: int,
) -> list[ChunkOut]:
    out: list[ChunkOut] = []
    for entry in toc:
        chapter = str(entry.get("title") or "Unknown Chapter").strip()
        p0 = int(entry.get("page_start") or 1)
        p1 = int(entry.get("page_end") or p0)
        chapter_pages = [p for p in pages if p0 <= p.page_number <= p1]
        combined = "\n".join(p.text for p in chapter_pages).strip()
        for piece in _chunk_text(combined, max_chars=max_chars, overlap_chars=overlap_chars):
            norm = _normalize_ws(piece)
            if len(norm) < 200:
                continue
            out.append(
                ChunkOut(
                    class_level=class_level,
                    subject=subject,
                    chapter=chapter,
                    page_start=p0,
                    page_end=p1,
                    text=norm,
                    text_hash=_hash_text(norm),
                    topics=_topics_from_text(piece),
                )
            )
    return out


def _chunk_by_heading(
    *,
    pages: list[PageText],
    class_level: int,
    subject: str,
    max_chars: int,
    overlap_chars: int,
) -> list[ChunkOut]:
    out: list[ChunkOut] = []
    current_chapter = "Unknown Chapter"
    current_page_start = pages[0].page_number if pages else 1
    current_page_end = current_page_start
    buffer: list[str] = []

    def flush():
        nonlocal buffer, current_page_start, current_page_end, current_chapter
        combined = "\n".join(buffer).strip()
        if not combined:
            buffer = []
            return
        for piece in _chunk_text(combined, max_chars=max_chars, overlap_chars=overlap_chars):
            norm = _normalize_ws(piece)
            if len(norm) < 200:
                continue
            out.append(
                ChunkOut(
                    class_level=class_level,
                    subject=subject,
                    chapter=current_chapter,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    text=norm,
                    text_hash=_hash_text(norm),
                    topics=_topics_from_text(piece),
                )
            )
        buffer = []

    for page in pages:
        detected = _detect_chapter_from_page(page.text)
        if detected and detected != current_chapter and buffer:
            flush()
            current_chapter = detected
            current_page_start = page.page_number
        current_page_end = page.page_number
        buffer.append(page.text)

    flush()
    return out



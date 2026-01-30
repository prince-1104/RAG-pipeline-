"""
Token-based chunking for offline textbook indexing (NCERT / CBSE tuned).

This module is used ONLY in the offline indexing pipeline:
- Input is extracted text (page-wise), not PDFs.
- Output is chunk strings ONLY (no metadata here).
- Deterministic and repeatable for the same input.

Core idea:
- Prefer paragraph boundaries (blank-line separated).
- Accumulate paragraphs into ~450–550 token chunks with a hard cap.
- If a paragraph is too large, fall back to sentence-level splitting.
- Never split mid-sentence unless unavoidable (single sentence exceeds max tokens).
"""

from __future__ import annotations

import re
from typing import Any

import tiktoken


# Deterministic tokenizer used across the pipeline.
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return token count using cl100k_base (deterministic)."""
    if not text:
        return 0
    return len(_ENC.encode(text))


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")  # blank line(s) with optional whitespace

# Sentence boundary heuristic:
# - Supports English-like endings (. ! ?) and Devanagari danda (।) common in Hindi NCERT.
# - We keep punctuation with the sentence to preserve meaning.
_SENTENCE_RE = re.compile(r".+?(?:[.!?]|।)(?=\s+|$)", flags=re.DOTALL)


def _normalize_text(s: str) -> str:
    """
    Normalize text defensively without changing semantic content too much:
    - Standardize newlines
    - Convert NBSP to space
    - Collapse 3+ blank lines into a single paragraph break
    """
    if not s:
        return ""
    s = s.replace("\u00A0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Reduce extreme whitespace-only gaps while preserving paragraph breaks.
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs, ignoring whitespace-only paragraphs."""
    text = _normalize_text(text)
    if not text:
        return []
    paras = [p.strip() for p in _PARA_SPLIT_RE.split(text)]
    return [p for p in paras if p]


def _split_into_sentences(paragraph: str) -> list[str]:
    """
    Best-effort sentence splitter.

    Notes:
    - This is intentionally heuristic (no heavy NLP deps).
    - If we fail to detect boundaries, we return the whole paragraph as one "sentence".
    """
    p = paragraph.strip()
    if not p:
        return []

    matches = [m.group(0).strip() for m in _SENTENCE_RE.finditer(p) if m.group(0).strip()]
    if not matches:
        return [p]

    # If regex didn't consume the full paragraph (e.g., missing terminal punctuation),
    # add the remainder as the last sentence.
    consumed = sum(len(m.group(0)) for m in _SENTENCE_RE.finditer(p))
    if consumed < len(p):
        tail = p[consumed:].strip()
        if tail:
            matches.append(tail)
    return matches


def _decode_last_tokens(text: str, n_tokens: int) -> str:
    """Decode the last N tokens of text (used for overlap)."""
    if n_tokens <= 0:
        return ""
    toks = _ENC.encode(text)
    if not toks:
        return ""
    return _ENC.decode(toks[-n_tokens:]).strip()


def _hard_token_split(text: str, max_tokens: int) -> list[str]:
    """
    Last-resort splitter: split by tokens and decode back to text.
    This can cut mid-sentence; we only use it when a *single sentence*
    is longer than max_tokens (unavoidable to enforce the hard cap).
    """
    t = text.strip()
    if not t:
        return []
    toks = _ENC.encode(t)
    if len(toks) <= max_tokens:
        return [t]
    out: list[str] = []
    for i in range(0, len(toks), max_tokens):
        piece = _ENC.decode(toks[i : i + max_tokens]).strip()
        if piece:
            out.append(piece)
    return out


def hard_token_split(text: str, max_tokens: int) -> list[str]:
    """Public wrapper for token-based hard splitting."""
    return _hard_token_split(text, max_tokens=max_tokens)


def _paragraph_to_units(paragraph: str, max_tokens: int) -> list[str]:
    """
    Convert a paragraph into one or more units where each unit <= max_tokens.

    - If paragraph fits, it remains intact (best semantic preservation).
    - If too big, we split into sentences and pack them into units.
    - If a single sentence is still too big, we fall back to hard token split.
    """
    p = paragraph.strip()
    if not p:
        return []

    if count_tokens(p) <= max_tokens:
        return [p]

    sentences = _split_into_sentences(p)
    units: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush_buf() -> None:
        nonlocal buf, buf_tokens
        if buf:
            unit = " ".join(buf).strip()
            if unit:
                units.append(unit)
        buf = []
        buf_tokens = 0

    for s in sentences:
        st = s.strip()
        if not st:
            continue
        st_tokens = count_tokens(st)

        if st_tokens > max_tokens:
            # Unavoidable: enforce max token cap even if it cuts the sentence.
            flush_buf()
            units.extend(_hard_token_split(st, max_tokens=max_tokens))
            continue

        if buf_tokens == 0:
            buf = [st]
            buf_tokens = st_tokens
            continue

        # +1-ish token for a joining space is model-dependent; use exact recount for safety.
        candidate = " ".join([*buf, st])
        if count_tokens(candidate) <= max_tokens:
            buf.append(st)
            buf_tokens = count_tokens(candidate)
        else:
            flush_buf()
            buf = [st]
            buf_tokens = st_tokens

    flush_buf()
    return [u for u in units if u and count_tokens(u) <= max_tokens]


def chunk_text(pages: list[dict], max_tokens: int = 500, overlap_tokens: int = 80) -> list[str]:
    """
    Chunk extracted text pages into token-bounded chunks (strings only).

    Expected input shape (defensive; extra keys are ignored):
    - pages: list of dicts like {"page_number": int, "chapter_id": str|int|None, "text": str}
      The chunker uses only:
        - chapter_id: to enforce chapter boundaries (never cross chapters)
        - text: page-wise extracted text

    Rules implemented:
    - Prefer paragraph boundaries (split on blank lines).
    - Target chunk size is ~450–550 tokens; hard cap is enforced by max_tokens (clamped to 550).
    - Overlap is applied using the LAST overlap_tokens tokens of the previous chunk to
      preserve continuity across chunk boundaries (helps embeddings later).
    - If a paragraph exceeds the cap, fall back to sentence splitting; only split mid-sentence
      when a single sentence itself exceeds max_tokens (unavoidable).
    - Do not cross chapter boundaries (chapter_id changes cause a flush with NO carryover overlap).
    - Ignore empty / whitespace-only paragraphs.
    """
    if not pages:
        return []

    # Enforce the "hard max 550" requirement defensively.
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    max_tokens = min(int(max_tokens), 550)

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    overlap_tokens = int(overlap_tokens)
    # Recommended overlap for retrieval embeddings is 60–80 tokens; enforce that range
    # when overlap is enabled, but still respect tiny max_tokens caps.
    if overlap_tokens != 0:
        overlap_tokens = max(60, min(overlap_tokens, 80))
    overlap_tokens = min(overlap_tokens, max_tokens - 1) if max_tokens > 1 else 0

    chunks: list[str] = []

    current_chapter: Any = None
    current_chunk = ""

    def flush_chunk() -> None:
        nonlocal current_chunk
        txt = current_chunk.strip()
        if txt:
            # Hard guarantee: never return a chunk above the cap.
            if count_tokens(txt) > max_tokens:
                # Defensive fallback (should be rare due to checks while building).
                for piece in _hard_token_split(txt, max_tokens=max_tokens):
                    if piece:
                        chunks.append(piece.strip())
            else:
                chunks.append(txt)
        current_chunk = ""

    for page in pages:
        # Defensive extraction of inputs (we expect keys: chapter_id, text).
        chapter_id = page.get("chapter_id")
        text = page.get("text") or ""

        # Do not cross chapter boundaries: flush and reset overlap carryover.
        if current_chapter is None:
            current_chapter = chapter_id
        elif chapter_id != current_chapter:
            flush_chunk()
            current_chapter = chapter_id

        for para in _split_paragraphs(str(text)):
            # Expand a paragraph into one or more <=max_tokens units.
            for unit in _paragraph_to_units(para, max_tokens=max_tokens):
                unit = unit.strip()
                if not unit:
                    continue

                if not current_chunk:
                    current_chunk = unit
                    continue

                candidate = f"{current_chunk}\n\n{unit}"
                if count_tokens(candidate) <= max_tokens:
                    current_chunk = candidate
                    continue

                # Exceeded cap: finalize current chunk and start next with overlap.
                flush_chunk()

                unit_tokens = count_tokens(unit)
                # Overlap is *token-based* (never character-based) to be model-aligned.
                # If the next unit is very large, we may reduce overlap so the new chunk
                # still respects the hard cap.
                allowed_overlap = max(0, max_tokens - unit_tokens)
                overlap_n = min(overlap_tokens, allowed_overlap)
                overlap_text = _decode_last_tokens(chunks[-1], overlap_n) if chunks else ""

                if overlap_text:
                    current_chunk = f"{overlap_text}\n\n{unit}"
                else:
                    current_chunk = unit

                # Final safety: ensure we never exceed max_tokens.
                if count_tokens(current_chunk) > max_tokens:
                    # Reduce overlap to fit (deterministic).
                    if overlap_text:
                        # Compute exact allowed overlap again using token lists.
                        prev_toks = _ENC.encode(chunks[-1])
                        overlap_toks = prev_toks[-overlap_n:] if overlap_n > 0 else []
                        # Drop tokens from the left of the overlap until it fits.
                        while overlap_toks:
                            trial = f"{_ENC.decode(overlap_toks).strip()}\n\n{unit}".strip()
                            if count_tokens(trial) <= max_tokens:
                                current_chunk = trial
                                break
                            overlap_toks = overlap_toks[1:]
                        else:
                            current_chunk = unit

                    # If it still doesn't fit (should only happen if unit itself is too big),
                    # hard-split unit and proceed.
                    if count_tokens(current_chunk) > max_tokens:
                        pieces = _hard_token_split(unit, max_tokens=max_tokens)
                        if pieces:
                            current_chunk = pieces[0]
                            for extra in pieces[1:]:
                                flush_chunk()
                                current_chunk = extra

    flush_chunk()
    return [c.strip() for c in chunks if c.strip()]


# Example usage (offline indexing pipeline):
#
# pages = [
#     {"page_number": 1, "chapter_id": "chem-01", "text": "Intro text...\n\nFirst paragraph..."},
#     {"page_number": 2, "chapter_id": "chem-01", "text": "Next page paragraph...\n\nAnother paragraph..."},
#     {"page_number": 3, "chapter_id": "chem-02", "text": "New chapter starts here...\n\n..."},
# ]
#
# chunks = chunk_text(pages, max_tokens=550, overlap_tokens=80)
# # -> list[str] (no metadata)


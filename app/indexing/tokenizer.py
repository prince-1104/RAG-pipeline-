from __future__ import annotations

import tiktoken

# Single shared tokenizer for all token counting/truncation in indexing.
_ENC = tiktoken.get_encoding("cl100k_base")


def encode(text: str) -> list[int]:
    if not text:
        return []
    return _ENC.encode(text)


def decode(tokens: list[int]) -> str:
    if not tokens:
        return ""
    return _ENC.decode(tokens)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _ENC.decode(toks[:max_tokens]).strip()

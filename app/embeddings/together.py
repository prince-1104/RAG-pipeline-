from __future__ import annotations

from typing import List

import requests

from app.embeddings.base import EmbeddingClient
from app.indexing import tokenizer
from app.observability.agentbill import finalize_event, start_event
from app.settings import get_settings


class TogetherEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.together_api_key
        self.model = settings.embedding_model
        self.url = "https://api.together.xyz/v1/embeddings"
        self.embedding_dim = settings.embedding_dim

        if not self.api_key or not str(self.api_key).strip():
            raise RuntimeError("missing_together_api_key")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts one at a time since Together's 512 token limit is per-request total."""
        if not texts:
            return []
        settings = get_settings()
        # Together's 512 token limit applies to the TOTAL tokens in the request.
        # We send one input at a time and cap each to ~200 tiktoken tokens (~500 Together tokens).
        hard_cap = min(int(settings.embedding_hard_cap_tokens), 200)
        from app.observability.logging import log

        def _cap_single(text: str, cap_tokens: int) -> tuple[str, int]:
            payload = text or ""
            token_count = tokenizer.count_tokens(payload)
            if token_count > cap_tokens:
                log.warning(
                    "embedding_payload_truncated",
                    provider="together",
                    original_tokens=token_count,
                    hard_cap=cap_tokens,
                )
                payload = tokenizer.truncate_to_tokens(payload, cap_tokens)
            return payload, tokenizer.count_tokens(payload)

        def _embed_single(text: str, cap_tokens: int) -> List[float]:
            """Embed a single text with retry logic for token limit errors."""
            max_retries = 3
            current_cap = cap_tokens
            payload, tok_count = _cap_single(text, current_cap)

            for attempt in range(1, max_retries + 1):
                resp = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": payload,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json().get("data")
                    if isinstance(data, list) and len(data) > 0:
                        vec = data[0].get("embedding")
                        if isinstance(vec, list):
                            vec_out = [float(x) for x in vec]
                            if self.embedding_dim and len(vec_out) != self.embedding_dim:
                                raise RuntimeError(
                                    f"embedding_dimension_mismatch expected={self.embedding_dim} got={len(vec_out)}"
                                )
                            return vec_out
                    raise RuntimeError("together_embedding_invalid_response_shape")

                body_preview = (resp.text or "")[:300].lower()
                is_context_error = (
                    resp.status_code == 400
                    and "maximum context length" in body_preview
                )
                if not is_context_error or attempt == max_retries:
                    raise RuntimeError(
                        f"Together embedding failed {resp.status_code}: {resp.text}"
                    )

                # Shrink cap and retry
                current_cap = max(50, int(current_cap * 0.6))
                log.warning(
                    "embedding_retry_with_lower_cap",
                    provider="together",
                    attempt=attempt,
                    new_hard_cap=current_cap,
                    status_code=resp.status_code,
                )
                payload, tok_count = _cap_single(text, current_cap)

            raise RuntimeError("together_embedding_max_retries_exceeded")

        event = start_event(
            event_name="ai_embedding",
            provider="together",
            model=self.model,
            kind="embedding",
        )
        total_input_tokens = 0
        out: List[List[float]] = []
        try:
            for text in texts:
                vec = _embed_single(text, hard_cap)
                out.append(vec)
                total_input_tokens += tokenizer.count_tokens(text or "")
            finalize_event(
                event,
                success=True,
                input_tokens=total_input_tokens,
                attributes={"ai.batch_size": len(texts)},
            )
        except Exception as exc:
            finalize_event(
                event,
                success=False,
                error=str(exc)[:100],
                input_tokens=total_input_tokens,
                attributes={"ai.batch_size": len(texts)},
            )
            raise
        return out

from __future__ import annotations

from typing import List

import requests

from app.embeddings.base import EmbeddingClient
from app.indexing import tokenizer
from app.observability.agentbill import finalize_event, start_event
from app.settings import get_settings


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.embedding_api_key or settings.openai_api_key
        self.model = settings.embedding_model
        self.base_url = settings.openai_base_url or "https://api.openai.com/v1"
        self.embedding_dim = settings.embedding_dim

        if not self.api_key or not str(self.api_key).strip():
            raise RuntimeError("missing_embedding_api_key")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        settings = get_settings()
        hard_cap = min(int(settings.embedding_hard_cap_tokens), 500)
        capped: List[str] = []
        total_input_tokens = 0
        for t in texts:
            payload = t or ""
            token_count = tokenizer.count_tokens(payload)
            if token_count > hard_cap:
                from app.observability.logging import log

                log.warning(
                    "embedding_payload_truncated",
                    provider="openai",
                    original_tokens=token_count,
                    hard_cap=hard_cap,
                )
                payload = tokenizer.truncate_to_tokens(payload, hard_cap)
            from app.observability.logging import log

            log.debug(
                "EMBED_DEBUG",
                provider="openai",
                token_count=tokenizer.count_tokens(payload),
                preview=payload[:200],
            )
            total_input_tokens += tokenizer.count_tokens(payload)
            capped.append(payload)

        event = start_event(
            event_name="ai_embedding",
            provider="openai",
            model=self.model,
            kind="embedding",
        )
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": capped,
            },
            timeout=60,
        )

        if resp.status_code != 200:
            finalize_event(
                event,
                success=False,
                error=f"openai_embedding_http_{resp.status_code}",
                input_tokens=total_input_tokens,
                attributes={"ai.batch_size": len(capped)},
            )
            raise RuntimeError(f"OpenAI embedding failed {resp.status_code}: {resp.text}")

        data = resp.json()
        items = data.get("data")
        if not isinstance(items, list):
            finalize_event(
                event,
                success=False,
                error="openai_embedding_invalid_response_shape",
                input_tokens=total_input_tokens,
                attributes={"ai.batch_size": len(capped)},
            )
            raise RuntimeError("openai_embedding_invalid_response_shape")

        by_idx: dict[int, List[float]] = {}
        for item in items:
            idx = item.get("index")
            vec = item.get("embedding")
            if idx is None or not isinstance(vec, list):
                raise RuntimeError("openai_embedding_invalid_item")
            by_idx[int(idx)] = [float(x) for x in vec]

        out: List[List[float]] = []
        for i in range(len(texts)):
            if i not in by_idx:
                finalize_event(
                    event,
                    success=False,
                    error=f"openai_embedding_missing_index:{i}",
                    input_tokens=total_input_tokens,
                    attributes={"ai.batch_size": len(capped)},
                )
                raise RuntimeError(f"openai_embedding_missing_index:{i}")
            vec = by_idx[i]
            if self.embedding_dim and len(vec) != self.embedding_dim:
                finalize_event(
                    event,
                    success=False,
                    error="embedding_dimension_mismatch",
                    input_tokens=total_input_tokens,
                    attributes={"ai.batch_size": len(capped)},
                )
                raise RuntimeError(
                    f"embedding_dimension_mismatch expected={self.embedding_dim} got={len(vec)}"
                )
            out.append(vec)
        finalize_event(
            event,
            success=True,
            input_tokens=total_input_tokens,
            attributes={"ai.batch_size": len(capped)},
        )
        return out

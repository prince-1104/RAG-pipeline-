from __future__ import annotations

from typing import List

import requests

from app.embeddings.base import EmbeddingClient
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

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI embedding failed {resp.status_code}: {resp.text}")

        data = resp.json()
        items = data.get("data")
        if not isinstance(items, list):
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
                raise RuntimeError(f"openai_embedding_missing_index:{i}")
            vec = by_idx[i]
            if self.embedding_dim and len(vec) != self.embedding_dim:
                raise RuntimeError(
                    f"embedding_dimension_mismatch expected={self.embedding_dim} got={len(vec)}"
                )
            out.append(vec)
        return out

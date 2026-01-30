from __future__ import annotations

from typing import List

import requests

from app.embeddings.base import EmbeddingClient
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
        if not texts:
            return []

        resp = requests.post(
            self.url,
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
            raise RuntimeError(
                f"Together embedding failed {resp.status_code}: {resp.text}"
            )

        data = resp.json().get("data")
        if not isinstance(data, list):
            raise RuntimeError("together_embedding_invalid_response_shape")

        out: List[List[float]] = []
        for item in data:
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("together_embedding_invalid_item")
            vec_out = [float(x) for x in vec]
            if self.embedding_dim and len(vec_out) != self.embedding_dim:
                raise RuntimeError(
                    f"embedding_dimension_mismatch expected={self.embedding_dim} got={len(vec_out)}"
                )
            out.append(vec_out)
        return out

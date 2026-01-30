from __future__ import annotations

from typing import List


class EmbeddingClient:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

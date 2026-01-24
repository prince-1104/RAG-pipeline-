from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.settings import get_settings


_MODEL: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        s = get_settings()
        _MODEL = SentenceTransformer(s.embedding_model_name)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Returns float32 embeddings normalized for cosine similarity (dot product).
    """
    model = get_embedding_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)



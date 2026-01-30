from __future__ import annotations

from app.embeddings.openai import OpenAIEmbeddingClient
from app.embeddings.together import TogetherEmbeddingClient
from app.settings import get_settings


def get_embedding_client():
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "together":
        return TogetherEmbeddingClient()

    if provider == "openai":
        return OpenAIEmbeddingClient()

    raise ValueError(f"unknown_embedding_provider:{provider}")

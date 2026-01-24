"""Offline ingestion and indexing for RAG (Qdrant + embeddings)."""

from rag.ingest.indexing_worker import (
    batch_upsert_points,
    build_qdrant_points,
    embed_chunks,
    run_indexing,
)

__all__ = [
    "embed_chunks",
    "build_qdrant_points",
    "batch_upsert_points",
    "run_indexing",
]

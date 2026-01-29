from __future__ import annotations

from typing import Iterable

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from app.db.models import Chunk, Corpus, Document
from app.observability.logging import log
from app.settings import get_settings

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 64


def _get_embedding_api_key() -> str:
    settings = get_settings()
    api_key = settings.embedding_api_key or settings.openai_api_key
    if not api_key or not str(api_key).strip():
        raise RuntimeError("missing_embedding_api_key")
    return str(api_key).strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    api_key = _get_embedding_api_key()
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code != 200:
        err_body = (resp.text or "")[:500]
        raise RuntimeError(f"embedding_api_failed status={resp.status_code} body={err_body}")
    data = resp.json()
    items = data.get("data")
    if not isinstance(items, list):
        raise RuntimeError("embedding_api_invalid_response_shape")
    # Preserve order by index
    by_idx: dict[int, list[float]] = {}
    for item in items:
        idx = item.get("index")
        vec = item.get("embedding")
        if idx is None or not isinstance(vec, list):
            raise RuntimeError("embedding_api_invalid_item")
        if len(vec) != EMBEDDING_DIM:
            raise RuntimeError(
                f"embedding_dimension_mismatch expected={EMBEDDING_DIM} got={len(vec)}"
            )
        by_idx[int(idx)] = [float(x) for x in vec]
    out: list[list[float]] = []
    for i in range(len(texts)):
        if i not in by_idx:
            raise RuntimeError(f"embedding_missing_index:{i}")
        out.append(by_idx[i])
    return out


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_s,
    )


def delete_document_points(*, document_id: str) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    flt = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(filter=flt),
    )


def _chunk_payload(
    *, chunk: Chunk, corpus: Corpus, document: Document
) -> dict[str, object]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "corpus_id": chunk.corpus_id,
        "tenant_id": corpus.tenant_id,
        "board": corpus.board,
        "syllabus": document.syllabus,
        "syllabus_version": document.syllabus_version,
        "class_level": chunk.class_level,
        "subject": chunk.subject,
        "chapter": chunk.chapter,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
    }


def _batch(iterable: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def upsert_chunks(
    *, chunks: list[Chunk], corpus: Corpus, document: Document
) -> int:
    if not chunks:
        return 0
    settings = get_settings()
    client = get_qdrant_client()
    total = 0
    for batch_chunks in _batch(chunks, EMBEDDING_BATCH_SIZE):
        vectors = embed_texts([c.text for c in batch_chunks])
        points: list[PointStruct] = []
        for chunk, vector in zip(batch_chunks, vectors):
            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector=vector,
                    payload=_chunk_payload(
                        chunk=chunk,
                        corpus=corpus,
                        document=document,
                    ),
                )
            )
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
    log.info(
        "qdrant_upsert_complete",
        document_id=document.id,
        corpus_id=document.corpus_id,
        chunks=len(chunks),
    )
    return total

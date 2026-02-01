from __future__ import annotations

import time
from typing import Iterable

from sqlmodel import select
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
)

from app.db.models import Chunk, Corpus, Document
from app.db.session import get_session
from app.embeddings.factory import get_embedding_client
from app.observability.logging import log
from app.settings import get_settings

DEFAULT_EMBEDDING_BATCH_SIZE = 24
EMBEDDING_MAX_CHARS = 8000


def _normalize_text(text: str) -> str:
    cleaned = text.encode("utf-8", "ignore").decode("utf-8").strip()
    if len(cleaned) > EMBEDDING_MAX_CHARS:
        cleaned = cleaned[:EMBEDDING_MAX_CHARS]
    return cleaned


def _embed_with_split(
    items: list[tuple[int, str]],
    *,
    client,
    provider: str,
) -> list[tuple[int, list[float]]]:
    if not items:
        return []
    texts = [t for _, t in items]
    try:
        vectors = client.embed(texts)
        return [(items[i][0], vec) for i, vec in enumerate(vectors)]
    except RuntimeError as exc:
        msg = str(exc)
        if " 400" in msg or "400:" in msg or "status=400" in msg:
            if len(items) == 1:
                log.warning(
                    "embedding_item_skipped",
                    provider=provider,
                    error=msg[:200],
                )
                return []
            mid = max(1, len(items) // 2)
            left = _embed_with_split(items[:mid], client=client, provider=provider)
            right = _embed_with_split(items[mid:], client=client, provider=provider)
            return left + right
        raise


def embed_texts(texts: list[str]) -> list[tuple[int, list[float]]]:
    if not texts:
        return []
    settings = get_settings()
    provider = settings.embedding_provider
    client = get_embedding_client()
    max_attempts = 3
    delay_s = 0.5

    items: list[tuple[int, str]] = []
    for idx, txt in enumerate(texts):
        cleaned = _normalize_text(txt)
        if not cleaned:
            continue
        items.append((idx, cleaned))

    if not items:
        return []

    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            results = _embed_with_split(items, client=client, provider=provider)
            duration_ms = int((time.perf_counter() - start) * 1000)
            log.info(
                "embedding_batch_complete",
                provider=provider,
                batch_size=len(items),
                embedded=len(results),
                duration_ms=duration_ms,
            )
            return results
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "embedding_batch_failed",
                provider=provider,
                batch_size=len(items),
                attempt=attempt,
                error=str(exc),
            )
            if attempt == max_attempts:
                raise
            time.sleep(delay_s)
            delay_s *= 2
    return []


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_s,
    )


def ensure_qdrant_payload_indexes() -> None:
    """
    Ensure Qdrant payload indexes exist for all filterable fields.
    Fail fast if we cannot create required indexes.
    """
    settings = get_settings()
    client = get_qdrant_client()

    required = {
        "document_id": PayloadSchemaType.KEYWORD,
        "corpus_id": PayloadSchemaType.KEYWORD,
        "tenant_id": PayloadSchemaType.KEYWORD,
        "board": PayloadSchemaType.KEYWORD,
        "syllabus": PayloadSchemaType.KEYWORD,
        "syllabus_version": PayloadSchemaType.KEYWORD,
        "class_level": PayloadSchemaType.INTEGER,
        "class": PayloadSchemaType.KEYWORD,
        "subject": PayloadSchemaType.KEYWORD,
        "chapter": PayloadSchemaType.KEYWORD,
        "chapter_number": PayloadSchemaType.INTEGER,
    }

    collection = client.get_collection(collection_name=settings.qdrant_collection)
    payload_schema = getattr(collection, "payload_schema", None) or {}
    existing = set(payload_schema.keys()) if isinstance(payload_schema, dict) else set()

    missing = {field: schema for field, schema in required.items() if field not in existing}
    for field_name, field_schema in missing.items():
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field_name,
            field_schema=field_schema,
        )
        log.info(
            "qdrant_payload_index_created",
            collection=settings.qdrant_collection,
            field_name=field_name,
            field_schema=str(field_schema),
        )

    if missing:
        # Re-check to fail fast if index creation did not stick
        collection = client.get_collection(collection_name=settings.qdrant_collection)
        payload_schema = getattr(collection, "payload_schema", None) or {}
        existing = set(payload_schema.keys()) if isinstance(payload_schema, dict) else set()
        still_missing = [field for field in required.keys() if field not in existing]
        if still_missing:
            raise RuntimeError(f"qdrant_payload_index_missing:{','.join(still_missing)}")

    vector_params = getattr(collection.config.params, "vectors", None)
    vector_size = None
    if hasattr(vector_params, "size"):
        vector_size = vector_params.size
    elif isinstance(vector_params, dict) and vector_params:
        any_params = next(iter(vector_params.values()))
        vector_size = getattr(any_params, "size", None)

    if vector_size and vector_size != settings.embedding_dim:
        raise RuntimeError(
            f"qdrant_vector_dim_mismatch expected={settings.embedding_dim} got={vector_size}"
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


def _extract_chapter_number(chapter: str | None) -> int | None:
    if not chapter:
        return None
    text = str(chapter)
    for token in text.replace(":", " ").replace(".", " ").split():
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                return None
    return None


def _chunk_payload(
    *, chunk: Chunk, corpus: Corpus, document: Document
) -> dict[str, object]:
    chapter_number = _extract_chapter_number(chunk.chapter)
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "corpus_id": chunk.corpus_id,
        "tenant_id": corpus.tenant_id,
        "board": corpus.board,
        "syllabus": document.syllabus,
        "syllabus_version": document.syllabus_version,
        "class_level": chunk.class_level,
        "class": str(chunk.class_level),
        "subject": chunk.subject,
        "chapter": chunk.chapter,
        "chapter_number": chapter_number,
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
    batch_size = max(1, int(getattr(settings, "embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)))
    if settings.embedding_provider == "together":
        batch_size = min(batch_size, 8)
    for batch_chunks in _batch(chunks, batch_size):
        vectors = embed_texts([c.text for c in batch_chunks])
        points: list[PointStruct] = []
        for idx, vector in vectors:
            chunk = batch_chunks[idx]
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
        if points:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
            total += len(points)
    log.info(
        "qdrant_upsert_complete",
        document_id=document.id,
        corpus_id=document.corpus_id,
        chunks=len(chunks),
    )
    return total


def upsert_document_chunks(
    *, document_id: str, corpus: Corpus, document: Document
) -> int:
    settings = get_settings()
    client = get_qdrant_client()
    batch_size = max(1, int(getattr(settings, "embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)))
    if settings.embedding_provider == "together":
        batch_size = min(batch_size, 8)
    total = 0
    last_id: str | None = None

    with get_session() as session:
        while True:
            q = select(Chunk).where(Chunk.document_id == document_id)
            if last_id:
                q = q.where(Chunk.id > last_id)
            q = q.order_by(Chunk.id).limit(batch_size)
            batch_chunks = session.exec(q).all()
            if not batch_chunks:
                break
            last_id = batch_chunks[-1].id
            vectors = embed_texts([c.text for c in batch_chunks])
            points: list[PointStruct] = []
            for idx, vector in vectors:
                chunk = batch_chunks[idx]
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
            if points:
                client.upsert(collection_name=settings.qdrant_collection, points=points)
                total += len(points)

    log.info(
        "qdrant_upsert_complete",
        document_id=document.id,
        corpus_id=document.corpus_id,
        chunks=total,
    )
    return total

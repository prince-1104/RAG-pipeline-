from __future__ import annotations

import time
from typing import Iterable

from sqlalchemy import select
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
from app.db.session import get_session_ctx
from app.embeddings.factory import get_embedding_client
from app.indexing.chunker import count_tokens, truncate_to_tokens
from app.observability.logging import log
from app.settings import get_settings

DEFAULT_EMBEDDING_BATCH_SIZE = 8
def _normalize_text(text: str, *, max_tokens: int) -> str:
    cleaned = text.encode("utf-8", "ignore").decode("utf-8").strip()
    if not cleaned:
        return ""
    if count_tokens(cleaned) > max_tokens:
        cleaned = truncate_to_tokens(cleaned, max_tokens)
    return cleaned


def _embed_with_split(
    items: list[tuple[int, str]],
    *,
    client,
    provider: str,
    hard_cap_tokens: int,
) -> list[tuple[int, list[float]]]:
    if not items:
        return []
    # Enforce hard cap on the FINAL payload we send to the embedding API.
    texts: list[str] = []
    for _, t in items:
        payload = t
        if count_tokens(payload) > hard_cap_tokens:
            payload = truncate_to_tokens(payload, hard_cap_tokens)
        texts.append(payload)
    try:
        vectors = client.embed(texts)
        return [(items[i][0], vec) for i, vec in enumerate(vectors)]
    except RuntimeError as exc:
        msg = str(exc)
        if " 400" in msg or "400:" in msg or "status=400" in msg:
            log.error(
                "embedding_bad_request",
                provider=provider,
                batch_size=len(items),
                error=msg[:200],
            )
            raise RuntimeError("embedding_bad_request") from exc
        raise


def embed_texts(texts: list[str]) -> list[tuple[int, list[float]]]:
    if not texts:
        return []
    settings = get_settings()
    provider = settings.embedding_provider
    client = get_embedding_client()

    items: list[tuple[int, str]] = []
    hard_cap_tokens = min(int(settings.embedding_hard_cap_tokens), 500)
    for idx, txt in enumerate(texts):
        cleaned = _normalize_text(txt, max_tokens=hard_cap_tokens)
        if not cleaned:
            continue
        items.append((idx, cleaned))

    if not items:
        return []

    batch_size = max(1, int(getattr(settings, "embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)))
    if provider == "together":
        batch_size = min(batch_size, 8)

    out: list[tuple[int, list[float]]] = []
    start = time.perf_counter()
    try:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_start = time.perf_counter()
            results = _embed_with_split(
                batch,
                client=client,
                provider=provider,
                hard_cap_tokens=hard_cap_tokens,
            )
            duration_ms = int((time.perf_counter() - batch_start) * 1000)
            log.info(
                "embedding_batch_complete",
                provider=provider,
                batch_size=len(batch),
                embedded=len(results),
                duration_ms=duration_ms,
            )
            out.extend(results)
        # Ensure stable ordering by original indices
        out.sort(key=lambda x: x[0])
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "embedding_batch_failed",
            provider=provider,
            batch_size=len(items),
            error=str(exc),
        )
        raise


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
        "chapter_name": PayloadSchemaType.KEYWORD,
        "language": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "difficulty_hint": PayloadSchemaType.KEYWORD,
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
    chapter_number = chunk.chapter_number or _extract_chapter_number(chunk.chapter)
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "corpus_id": chunk.corpus_id,
        "tenant_id": corpus.tenant_id,
        "board": chunk.board or document.board or corpus.board,
        "syllabus": document.syllabus,
        "syllabus_version": document.syllabus_version,
        "class_level": chunk.class_level,
        "class": str(chunk.class_level),
        "subject": chunk.subject,
        "chapter": chunk.chapter,
        "chapter_number": chapter_number,
        "chapter_name": chunk.chapter_name or chunk.chapter,
        "language": chunk.language or document.language,
        "source_type": chunk.source_type or document.source_type,
        "difficulty_hint": chunk.difficulty_hint or document.difficulty_hint,
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


async def upsert_document_chunks(
    *, document_id: str, corpus: Corpus, document: Document
) -> int:
    settings = get_settings()
    client = get_qdrant_client()
    batch_size = max(1, int(getattr(settings, "embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)))
    if settings.embedding_provider == "together":
        batch_size = min(batch_size, 8)
    total = 0
    last_id: str | None = None

    async with get_session_ctx() as session:
        while True:
            q = select(Chunk).where(Chunk.document_id == document_id)
            if last_id:
                q = q.where(Chunk.id > last_id)
            q = q.order_by(Chunk.id).limit(batch_size)
            batch_chunks = (await session.execute(q)).scalars().all()
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

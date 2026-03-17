from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.indexing.qdrant_index import embed_texts, get_qdrant_client
from app.observability.logging import log
from app.settings import get_settings


@dataclass
class SearchChunkResult:
    chunk_id: str
    text: str
    score: float
    document_id: str | None = None
    corpus_id: str | None = None


def search_chunks(
    *,
    query: str,
    tenant_id: str | None = None,
    corpus_id: str | None = None,
    top_k: int = 10,
    strict_metadata_filter: bool = False,
    class_level: Optional[int] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
) -> list[SearchChunkResult]:
    """
    Qdrant similarity search with mandatory tenant/corpus isolation filters.

    When strict_metadata_filter=False (school search), only tenant_id and corpus_id
    are applied so uploads with NULL/General metadata still return hits.
    When strict_metadata_filter=True (admin), class_level/subject/chapter are added.
    """
    settings = get_settings()
    client = get_qdrant_client()

    vectors = embed_texts([query])
    if not vectors:
        return []
    vector = vectors[0][1]

    must_filters: list[FieldCondition] = []
    if tenant_id:
        must_filters.append(FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id))))
    if corpus_id:
        must_filters.append(FieldCondition(key="corpus_id", match=MatchValue(value=str(corpus_id))))
    if strict_metadata_filter:
        if class_level is not None:
            must_filters.append(FieldCondition(key="class_level", match=MatchValue(value=class_level)))
        if subject:
            must_filters.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
        if chapter:
            must_filters.append(FieldCondition(key="chapter", match=MatchValue(value=chapter)))

    qdrant_filter = Filter(must=must_filters) if must_filters else None

    def _do_search(collection: str, flt: Filter | None, limit: int) -> list:
        if hasattr(client, "search"):
            return client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=flt,
                limit=limit,
            )
        if hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=flt,
                limit=limit,
            )
            if hasattr(resp, "points"):
                return list(resp.points or [])
            if hasattr(resp, "result"):
                return list(resp.result or [])
            return list(resp or [])
        raise AttributeError("QdrantClient has no search or query_points")

    hits = _do_search(settings.qdrant_collection, qdrant_filter, max(1, min(top_k, 100)))

    results: list[SearchChunkResult] = []
    for h in hits or []:
        payload = h.payload or {}
        text = payload.get("text", "")
        if not text:
            continue
        results.append(
            SearchChunkResult(
                chunk_id=payload.get("chunk_id", ""),
                text=text,
                score=float(h.score),
                document_id=payload.get("document_id"),
                corpus_id=payload.get("corpus_id"),
            )
        )

    log.info(
        "school_search_chunks",
        tenant_id=tenant_id,
        corpus_id=corpus_id,
        query_len=len(query),
        top_k=top_k,
        hits=len(results),
    )
    return results

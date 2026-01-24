"""
Offline RAG indexing worker for the Niyukti platform.

- Generates embeddings for text chunks via OpenAI Embeddings API (1536 dim).
- Inserts points into Qdrant Cloud in batches (upsert, idempotent).
- Deterministic UUIDv5 IDs; no R2 access, no runtime embedding outside this pipeline.

Expects:
- chunker.py (text extraction/chunking) and Qdrant collection already exist.
- Env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_API_KEY.

Worker accepts: doc_id, board, class (as class_name), subject, chapter_number,
chapter_name, chunks. Payload stores "class" from class_name.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-ada-002"  # 1536 dimensions; same for index & query
EMBEDDING_DIM = 1536
BATCH_SIZE = 64  # 50–100 per spec
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "niyukti.rag")

logger = logging.getLogger(__name__)


def _get_env(key: str) -> str:
    """Read required env var; raise if missing."""
    v = os.environ.get(key)
    if not v or not str(v).strip():
        raise RuntimeError(f"Missing required environment variable: {key}")
    return str(v).strip()


# -----------------------------------------------------------------------------
# 1. Embed chunks (OpenAI API)
# -----------------------------------------------------------------------------


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Generate 1536-dim embeddings for each chunk using the OpenAI Embeddings API.

    - Uses EMBEDDING_API_KEY from environment.
    - Model: text-embedding-ada-002 (same for indexing and querying).
    - If the API call fails, raises immediately; no partial results.

    Returns:
        List of embedding vectors, one per chunk, each of length 1536.
    """
    if not chunks:
        return []

    api_key = _get_env("EMBEDDING_API_KEY")
    # OpenAI allows batches; we send all chunks in one request (within token limits).
    # For very large jobs, caller can chunk the chunk list and call embed_chunks per batch.
    payload: dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "input": chunks,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            OPENAI_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if resp.status_code != 200:
        err_body = resp.text[:500] if resp.text else ""
        raise RuntimeError(
            f"Embedding API failed: status={resp.status_code} body={err_body}"
        )

    data = resp.json()
    embs_raw = data.get("data")
    if not isinstance(embs_raw, list):
        raise RuntimeError("Embedding API returned unexpected shape: missing 'data' list")

    # Preserve order by index (API may return sorted by index field)
    by_idx: dict[int, list[float]] = {}
    for item in embs_raw:
        idx = item.get("index")
        vec = item.get("embedding")
        if idx is None or not isinstance(vec, list):
            raise RuntimeError("Embedding API returned invalid item")
        by_idx[int(idx)] = [float(x) for x in vec]

    out: list[list[float]] = []
    for i in range(len(chunks)):
        if i not in by_idx:
            raise RuntimeError(f"Embedding API did not return embedding for chunk index {i}")
        v = by_idx[i]
        if len(v) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(v)}"
            )
        out.append(v)

    return out


# -----------------------------------------------------------------------------
# 2. Build Qdrant points (deterministic IDs, payload)
# -----------------------------------------------------------------------------


def build_qdrant_points(
    *,
    doc_id: str,
    board: str,
    class_name: str,
    subject: str,
    chapter_number: str | int,
    chapter_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> list[qdrant_models.PointStruct]:
    """
    Build Qdrant point structs with deterministic UUIDv5 IDs and full payload.

    - ID seed: ``{doc_id}_{chunk_index}`` (namespace: niyukti.rag).
    - Payload: doc_id, board, class, subject, chapter_number, chapter_name,
      chunk_index, text.

    Raises:
        ValueError: if ``len(chunks) != len(embeddings)``.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )

    points: list[qdrant_models.PointStruct] = []
    for i, (text, vec) in enumerate(zip(chunks, embeddings)):
        seed = f"{doc_id}_{i}"
        point_id = str(uuid.uuid5(UUID_NAMESPACE, seed))
        payload = {
            "doc_id": doc_id,
            "board": board,
            "class": class_name,
            "subject": subject,
            "chapter_number": chapter_number if isinstance(chapter_number, int) else str(chapter_number),
            "chapter_name": chapter_name,
            "chunk_index": i,
            "text": text,
        }
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            )
        )
    return points


# -----------------------------------------------------------------------------
# 3. Batch upsert into Qdrant (with retry)
# -----------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Retry on network errors or Qdrant 5xx. Do not retry on 4xx (e.g. auth, validation)."""
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, UnexpectedResponse):
        sc = getattr(exc, "status_code", None)
        return sc is not None and sc >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _upsert_batch(
    client: QdrantClient,
    collection: str,
    batch: list[qdrant_models.PointStruct],
) -> None:
    """Execute a single upsert batch. Used internally with retry."""
    client.upsert(collection_name=collection, points=batch)


def batch_upsert_points(points: list[qdrant_models.PointStruct]) -> None:
    """
    Upsert points into Qdrant in batches of 50–100 (configurable via BATCH_SIZE).

    - Uses QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION from environment.
    - Idempotent (upsert). On Qdrant failure, retries with exponential backoff.
    - Does not swallow errors; raises after retries exhausted.
    """
    if not points:
        return

    url = _get_env("QDRANT_URL")
    api_key = _get_env("QDRANT_API_KEY")
    collection = _get_env("QDRANT_COLLECTION")

    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30.0,
    )

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        _upsert_batch(client, collection, batch)
        logger.info("qdrant_batch_upserted batch_start=%s batch_size=%s", i, len(batch))


# -----------------------------------------------------------------------------
# 4. Orchestrator: run full indexing for one document
# -----------------------------------------------------------------------------


def run_indexing(
    *,
    doc_id: str,
    board: str,
    class_name: str,
    subject: str,
    chapter_number: str | int,
    chapter_name: str,
    chunks: list[str],
) -> dict[str, Any]:
    """
    Run the full offline indexing pipeline for one document.

    1. Embed all chunks (fail fast on embedding error).
    2. Build Qdrant points with deterministic IDs.
    3. Batch upsert into Qdrant (retry on transient failures).

    Returns:
        {"status": "success", "chunks_indexed": int}.
    """
    if not chunks:
        logger.info("indexing_skipped_no_chunks doc_id=%s", doc_id)
        return {"status": "success", "chunks_indexed": 0}

    embeddings = embed_chunks(chunks)
    points = build_qdrant_points(
        doc_id=doc_id,
        board=board,
        class_name=class_name,
        subject=subject,
        chapter_number=chapter_number,
        chapter_name=chapter_name,
        chunks=chunks,
        embeddings=embeddings,
    )
    batch_upsert_points(points)

    n = len(chunks)
    logger.info("indexing_complete doc_id=%s chunks_indexed=%s", doc_id, n)
    return {"status": "success", "chunks_indexed": n}


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
#
# """Index chunks for a single chapter (chunks from chunker, etc.)."""
#
# import logging
# import os
#
# logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
#
# os.environ["QDRANT_URL"] = "https://xxx.qdrant.io"
# os.environ["QDRANT_API_KEY"] = "your-qdrant-api-key"
# os.environ["QDRANT_COLLECTION"] = "niyukti_knowledge"
# os.environ["EMBEDDING_API_KEY"] = "sk-..."
#
# from rag.ingest.indexing_worker import run_indexing
#
# result = run_indexing(
#     doc_id="ncert-math-10-ch1",
#     board="CBSE",
#     class_name="10",   # "class" in payload
#     subject="Mathematics",
#     chapter_number=1,
#     chapter_name="Real Numbers",
#     chunks=[
#         "Euclid's division lemma states that...",
#         "The fundamental theorem of arithmetic...",
#     ],
# )
# # -> {"status": "success", "chunks_indexed": 2}
#

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.settings import get_settings


@dataclass
class IndexMetadata:
    corpus_id: str
    class_level: int
    subject: str
    dim: int
    # vector_id -> chunk_id mapping (stored as list aligned to vectors)
    chunk_ids: list[str]
    # optional extra for debugging
    created_at_iso: str


def _get_index_dir(meta: IndexMetadata) -> Path:
    """Get the directory path for this index (local mode)."""
    s = get_settings()
    safe_subject = meta.subject.lower().replace(" ", "_")
    base = Path(s.index_base_path).resolve()
    if not base.is_absolute():
        # Relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        base = repo_root / base
    return base / meta.corpus_id / f"class_{meta.class_level}" / safe_subject


def _index_path(meta: IndexMetadata) -> Path:
    return _get_index_dir(meta) / "index.faiss"


def _meta_path(meta: IndexMetadata) -> Path:
    return _get_index_dir(meta) / "meta.json"


def _vecs_path(meta: IndexMetadata) -> Path:
    return _get_index_dir(meta) / "vecs.npy"


async def save_index(meta: IndexMetadata, vectors: np.ndarray) -> None:
    """
    Persist FAISS index + metadata to local disk under INDEX_BASE_PATH.
    We store vectors separately to enable metadata pre-filtered search subsets.
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # serialize
    faiss_array = faiss.serialize_index(index)  # Returns numpy array, not bytes
    meta_payload = json.dumps(meta.__dict__).encode("utf-8")
    vecs_buf = io.BytesIO()
    np.save(vecs_buf, vectors)
    vecs_bytes = vecs_buf.getvalue()
    
    # Save FAISS index as numpy array
    idx_buf = io.BytesIO()
    np.save(idx_buf, faiss_array)
    idx_bytes = idx_buf.getvalue()

    # Persist to local disk
    index_dir = _get_index_dir(meta)
    index_dir.mkdir(parents=True, exist_ok=True)

    _index_path(meta).write_bytes(idx_bytes)
    _meta_path(meta).write_bytes(meta_payload)
    _vecs_path(meta).write_bytes(vecs_bytes)


async def load_index(corpus_id: str, class_level: int, subject: str) -> tuple[IndexMetadata, np.ndarray, faiss.Index]:
    """
    Load FAISS index from local disk.
    Path uses normalized subject (lowercase, underscores), but metadata stores original subject name.
    """
    # Create temp metadata for path construction (path normalization happens in _get_index_dir)
    meta = IndexMetadata(corpus_id=corpus_id, class_level=class_level, subject=subject, dim=0, chunk_ids=[], created_at_iso="")
    
    meta_path = _meta_path(meta)
    if not meta_path.exists():
        # Provide helpful error message with available subjects
        base_dir = _get_index_dir(meta).parent
        if base_dir.exists():
            existing_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
            normalized_requested = subject.lower().replace(" ", "_")
            raise FileNotFoundError(
                f"index_not_found: {meta_path}\n"
                f"Requested: class_{class_level}/{normalized_requested}\n"
                f"Available subjects for class_{class_level}: {existing_dirs}"
            )
        raise FileNotFoundError(f"index_not_found: {meta_path}")
    
    meta_obj = IndexMetadata(**json.loads(meta_path.read_bytes().decode("utf-8")))
    
    vecs_path = _vecs_path(meta_obj)
    if not vecs_path.exists():
        raise FileNotFoundError(f"vectors_not_found: {vecs_path}")
    vectors = np.load(io.BytesIO(vecs_path.read_bytes())).astype(np.float32)
    
    idx_path = _index_path(meta_obj)
    if not idx_path.exists():
        raise FileNotFoundError(f"faiss_index_not_found: {idx_path}")
    # Load FAISS index: support both old format (bytes) and new format (numpy array)
    idx_bytes = idx_path.read_bytes()
    try:
        # Try new format: numpy array saved with np.save()
        faiss_array = np.load(io.BytesIO(idx_bytes), allow_pickle=False)
        index = faiss.deserialize_index(faiss_array)
    except (ValueError, TypeError):
        # Fallback to old format: assume it's already a serialized numpy array as bytes
        # Try to deserialize directly (for backward compatibility with old indices)
        try:
            index = faiss.deserialize_index(idx_bytes)
        except Exception:
            # Last resort: try converting bytes to numpy array
            faiss_array = np.frombuffer(idx_bytes, dtype=np.uint8)
            index = faiss.deserialize_index(faiss_array)
    
    return meta_obj, vectors, index



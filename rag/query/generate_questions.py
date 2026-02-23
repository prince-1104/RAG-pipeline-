"""
Runtime RAG API handler for the Niyukti platform.

Retrieves relevant chunks from Qdrant, generates assessment questions via LLM,
and returns structured questions. HOT PATH: fast and reliable.

- Uses same embedding model as indexing (OpenAI text-embedding-ada-002).
- Qdrant collection: niyukti_knowledge. Strict filters: board, class, subject, chapter_number.
- No R2 reads, no re-embedding of chunks, no Qdrant writes.
- Errors are propagated; never swallowed.

Example request:
  POST /rag/generate/questions
  {
    "board": "CBSE",
    "class": 10,
    "subject": "Science",
    "chapter_number": 2,
    "difficulty": "medium",
    "question_type": "mcq",
    "count": 10
  }

Example response:
  {
    "questions": [
      {
        "stem": "What is the chemical formula of water?",
        "options": ["H2O", "CO2", "NaCl", "O2"],
        "answer": "H2O",
        "qtype": "mcq",
        "difficulty": "medium"
      }
    ]
  }
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.embeddings.factory import get_embedding_client
from app.llm.gemini import GeminiClient
from app.observability.logging import log
from app.observability.agentbill import finalize_event, start_event
from app.settings import get_settings


TOP_K_MIN = 12
TOP_K_MAX = 20
TOP_K_DEFAULT = 16

Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["mcq", "descriptive"]

RAG_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stem": {"type": "string"},
                    "options": {"type": ["array", "null"], "items": {"type": "string"}},
                    "answer": {"type": "string"},
                    "qtype": {"type": "string"},
                    "difficulty": {"type": "string"},
                },
                "required": ["stem", "answer", "qtype", "difficulty"],
            },
        },
        "note": {"type": "string"},
    },
    "required": ["questions"],
}


# -----------------------------------------------------------------------------
# Embedding (single query; same model as indexing)
# -----------------------------------------------------------------------------


def embed_query(query: str) -> list[float]:
    """
    Generate a single query embedding via configured provider.

    Raises:
        RuntimeError: on embedding failure.
    """
    client = get_embedding_client()
    settings = get_settings()
    vectors = client.embed([query])
    if len(vectors) != 1:
        raise RuntimeError("embedding_query_unexpected_vector_count")
    vec = vectors[0]
    if settings.embedding_dim and len(vec) != settings.embedding_dim:
        raise RuntimeError(
            f"embedding_dimension_mismatch expected={settings.embedding_dim} got={len(vec)}"
        )
    return vec


# -----------------------------------------------------------------------------
# Retrieval (Qdrant search with strict filters)
# -----------------------------------------------------------------------------


def _qdrant_search(
    client: QdrantClient, *, collection: str, vector: list[float], flt: Filter, limit: int
) -> list[Any]:
    """
    Compatibility wrapper for qdrant-client API differences.
    Returns list of scored points.
    """
    if hasattr(client, "search_points"):
        try:
            resp = client.search_points(
                collection_name=collection,
                vector=vector,
                filter=flt,
                limit=limit,
                with_payload=True,
            )
        except TypeError:
            resp = client.search_points(
                collection_name=collection,
                query_vector=vector,
                query_filter=flt,
                limit=limit,
            )
        return resp.points if hasattr(resp, "points") else resp
    if hasattr(client, "query_points"):
        try:
            resp = client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        except TypeError:
            resp = client.query_points(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        return resp.points if hasattr(resp, "points") else resp
    return client.search(
        collection_name=collection,
        query_vector=vector,
        query_filter=flt,
        limit=limit,
    )


def retrieve_chunks(
    *,
    board: str,
    class_val: int,
    subject: str,
    chapter_number: int,
    query: str,
    top_k: int = TOP_K_DEFAULT,
) -> list[str]:
    """
    Query Qdrant with strict filters (board, class, subject, chapter_number),
    cosine similarity, top_k results. Returns list of 'text' from payloads only.

    Raises:
        RuntimeError: on Qdrant or embedding errors.
    """
    k = max(TOP_K_MIN, min(TOP_K_MAX, top_k))
    vector = embed_query(query)

    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_s,
    )

    # Payload: board, class_level (int), subject, chapter_number (int).
    filter_clause = Filter(
        must=[
            FieldCondition(key="board", match=MatchValue(value=board)),
            FieldCondition(key="class_level", match=MatchValue(value=class_val)),
            FieldCondition(key="subject", match=MatchValue(value=subject)),
            FieldCondition(key="chapter_number", match=MatchValue(value=chapter_number)),
        ]
    )

    hits = _qdrant_search(
        client,
        collection=settings.qdrant_collection,
        vector=vector,
        flt=filter_clause,
        limit=k,
    )

    if not hits:
        # Fallback: relax chapter filter to avoid empty results when chapter parsing is missing.
        fallback_filter = Filter(
            must=[
                FieldCondition(key="board", match=MatchValue(value=board)),
                FieldCondition(key="class_level", match=MatchValue(value=class_val)),
                FieldCondition(key="subject", match=MatchValue(value=subject)),
            ]
        )
        hits = _qdrant_search(
            client,
            collection=settings.qdrant_collection,
            vector=vector,
            flt=fallback_filter,
            limit=k,
        )

    texts: list[str] = []
    for h in hits:
        payload = h.payload or {}
        t = payload.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())

    log.info(
        "rag_retrieve_chunks",
        board=board,
        class_val=class_val,
        subject=subject,
        chapter_number=chapter_number,
        requested_k=k,
        hits=len(hits),
        texts_extracted=len(texts),
    )
    return texts


# -----------------------------------------------------------------------------
# Prompt and LLM
# -----------------------------------------------------------------------------


def build_prompt(
    chunks: list[str],
    *,
    difficulty: Difficulty,
    question_type: QuestionType,
    count: int,
) -> tuple[str, str]:
    """
    Build system and user prompts. Context = retrieved chunks only.
    Explicit: no hallucination; if answer not in context, say so. JSON only.
    """
    context = "\n\n---\n\n".join(chunks) if chunks else "(No context retrieved.)"

    system = """You are an assessment question generator for the Niyukti platform.
Use ONLY the provided context to generate questions. Do not invent facts.
If the answer to a question is not present in the context, say so explicitly.
Output valid JSON only. No markdown, no explanation outside JSON."""

    user = f"""Context from the textbook:

{context}

Generate exactly {count} {question_type} question(s), each of {difficulty} difficulty.
Output a single JSON object with a key "questions" (array of question objects).
Each question must have:
- "stem": string (question text)
- "options": array of strings (only for mcq; null for descriptive)
- "answer": string
- "qtype": "{question_type}"
- "difficulty": "{difficulty}"

If the context is insufficient to create {count} questions, create fewer and say so in a "note" field.
Output nothing but the JSON object."""

    return system, user


def generate_questions(
    *,
    board: str,
    class_val: int,
    subject: str,
    chapter_number: int,
    difficulty: Difficulty,
    question_type: QuestionType,
    count: int,
) -> list[dict[str, Any]]:
    """
    Full pipeline: semantic query -> embed -> retrieve -> prompt -> LLM -> parse.
    Returns list of structured question dicts. Never returns raw LLM text.
    Raises on validation, retrieval, or LLM errors.
    """
    query = (
        f"Board {board} Class {class_val} Subject {subject} Chapter {chapter_number}. "
        f"Key concepts and content for {question_type} questions at {difficulty} difficulty."
    )
    chunks = retrieve_chunks(
        board=board,
        class_val=class_val,
        subject=subject,
        chapter_number=chapter_number,
        query=query,
        top_k=TOP_K_DEFAULT,
    )

    if not chunks:
        log.warning(
            "rag_no_chunks",
            board=board,
            class_val=class_val,
            subject=subject,
            chapter_number=chapter_number,
        )
        return []

    system, user = build_prompt(
        chunks, difficulty=difficulty, question_type=question_type, count=count
    )

    settings = get_settings()
    openai_api_key = getattr(settings, "openai_api_key", None)
    openrouter_api_key = getattr(settings, "openrouter_api_key", None)
    llm_api_key = getattr(settings, "llm_api_key", None)
    gemini_api_key = getattr(settings, "gemini_api_key", None)

    def _call_openai(
        *,
        api_key: str,
        model: str,
        base_url: str,
        provider: str,
    ) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        event = start_event(
            event_name="ai_completion",
            provider=provider,
            model=model,
            kind="completion",
        )
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
            )
        if resp.status_code != 200:
            finalize_event(
                event,
                success=False,
                error=f"{provider}_http_{resp.status_code}",
            )
            err_body = (resp.text or "")[:500]
            raise RuntimeError(
                f"LLM API failed: status={resp.status_code} body={err_body}"
            )
        data = resp.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        input_tokens = None
        output_tokens = None
        total_tokens = None
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            finalize_event(
                event,
                success=False,
                error=f"{provider}_missing_content",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            raise RuntimeError(f"LLM response missing content: {e}") from e
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            finalize_event(
                event,
                success=False,
                error=f"{provider}_invalid_json",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            raise RuntimeError(f"LLM did not return valid JSON: {e}") from e
        finalize_event(
            event,
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            attributes={"ai.response_id": data.get("id")},
        )
        return parsed

    def _call_gemini(*, api_key: str, model: str) -> dict[str, Any]:
        client = GeminiClient(api_key=api_key, model=model)
        result = client.generate_json(
            system_prompt=system,
            user_prompt=user,
            json_schema=RAG_QUESTION_SCHEMA,
        )
        if not isinstance(result.parsed_json, dict):
            raise RuntimeError("gemini_invalid_json_output: expected object")
        return result.parsed_json

    parsed: dict[str, Any] | None = None
    errors: list[str] = []

    # Primary: OpenAI (direct)
    if openai_api_key and str(openai_api_key).strip():
        try:
            parsed = _call_openai(
                api_key=openai_api_key,
                model=getattr(settings, "openai_model", "gpt-4o-mini"),
                base_url=getattr(settings, "openai_base_url", None)
                or "https://api.openai.com/v1",
                provider="openai",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openai_failed:{str(exc)[:200]}")
            log.warning("rag_openai_failed_fallback", error=str(exc))

    # Secondary: Gemini fallback
    if parsed is None and gemini_api_key and str(gemini_api_key).strip():
        try:
            parsed = _call_gemini(
                api_key=gemini_api_key,
                model=getattr(settings, "gemini_model", "gemini-2.5-flash"),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gemini_failed:{str(exc)[:200]}")
            log.error("rag_gemini_failed", error=str(exc))

    # Optional: OpenRouter/LLM API key (legacy)
    if parsed is None:
        api_key = (llm_api_key or openrouter_api_key or "").strip()
        if api_key:
            try:
                parsed = _call_openai(
                    api_key=api_key,
                    model=getattr(settings, "llm_model", None)
                    or getattr(settings, "openrouter_model", "openai/gpt-4o-mini"),
                    base_url=getattr(settings, "llm_base_url", None)
                    or getattr(settings, "openrouter_base_url", None)
                    or "https://openrouter.ai/api/v1",
                    provider="openrouter",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"openrouter_failed:{str(exc)[:200]}")
                log.error("rag_openrouter_failed", error=str(exc))

    if parsed is None:
        raise RuntimeError(
            "llm_api_key or LLM_API_KEY (or openai/gemini) required for runtime RAG; "
            f"errors={errors}"
        )

    questions = parsed.get("questions")
    if not isinstance(questions, list):
        raise RuntimeError(
            f"LLM JSON must contain 'questions' array; got {type(questions).__name__}"
        )

    out: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            log.warning("rag_skip_non_dict_question", index=i)
            continue
        stem = q.get("stem")
        if not isinstance(stem, str) or not stem.strip():
            log.warning("rag_skip_missing_stem", index=i)
            continue
        out.append({
            "stem": stem.strip(),
            "options": q.get("options") if question_type == "mcq" else None,
            "answer": (q.get("answer") or "").strip() or None,
            "qtype": question_type,
            "difficulty": difficulty,
        })

    log.info(
        "rag_generate_complete",
        board=board,
        class_val=class_val,
        subject=subject,
        chapter_number=chapter_number,
        requested=count,
        returned=len(out),
    )
    return out

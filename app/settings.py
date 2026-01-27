from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    service_name: str = "niyukti-question-gen"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8080

    database_url: str = "postgresql://localhost/niyukti_qgen"

    # Storage (local PDF mode or R2)
    storage_backend: Literal["local", "r2"] = "local"
    pdf_base_path: str = "data/pdfs"
    index_base_path: str = "data/indexes"
    uploads_base_path: str = ".data/uploads"
    
    # Cloudflare R2 configuration (S3-compatible)
    r2_account_id: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket_name: str | None = None
    r2_endpoint_url: str | None = None  # e.g., https://<account-id>.r2.cloudflarestorage.com
    r2_region: str = "auto"

    # Qdrant vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "niyukti_knowledge"
    qdrant_timeout_s: float = 30.0

    # Redis configuration (for question pool caching)
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str | None = None
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "stub"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"  # Updated to newer model; falls back to stub if unavailable
    
    # OpenAI (direct) configuration
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None  # Optional; defaults to https://api.openai.com/v1

    # OpenRouter configuration (unified API for multiple LLM providers)
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"  # Default: cost-effective OpenAI model via OpenRouter
    openrouter_base_url: str | None = None  # Optional: defaults to https://openrouter.ai/api/v1

    # Runtime RAG (Qdrant + LLM): env EMBEDDING_API_KEY, LLM_API_KEY, LLM_MODEL
    embedding_api_key: str | None = None  # OpenAI embeddings; same as indexing
    llm_api_key: str | None = None  # OpenAI / OpenRouter for question generation
    llm_model: str = "gpt-4o-mini"  # e.g. gpt-4o-mini or openai/gpt-4o-mini via OpenRouter
    llm_base_url: str | None = None  # Optional; defaults to OpenRouter

    requests_per_minute: int = 120
    max_context_chunks: int = 12
    min_retrieval_confidence: float = 0.55

    # LLM performance tuning
    llm_timeout_s: float = 90.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    # Pool generation performance tuning
    pool_generation_max_concurrency: int = 6
    pool_generation_context_chunks: int = 6
    pool_generation_chunk_chars: int = 900
    pool_generation_min_absolute: int = 10
    pool_generation_min_fraction: float = 0.2

    # Admin generation performance tuning (per-class parallelism)
    admin_generate_max_concurrency: int = 2

    # Debug logging (file I/O can slow generation; keep off by default)
    debug_log_enabled: bool = False

    # API Authentication (optional)
    api_key: str | None = None  # API key for service authentication
    api_keys: str | None = None  # Comma-separated list of valid API keys


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    # In dev, reload settings each call so `.env` edits take effect without
    # requiring a full server restart (useful during local integration/debugging).
    if _SETTINGS is None:
        _SETTINGS = Settings()
        return _SETTINGS

    if _SETTINGS.env == "dev":
        _SETTINGS = Settings()

    return _SETTINGS



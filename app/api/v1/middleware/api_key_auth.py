from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.responses import ORJSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from app.settings import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API keys from X-API-Key header"""

    def _extract_api_key(self, request: Request) -> str | None:
        """
        Accept API key in either:
        - X-API-Key: <key>
        - Authorization: Bearer <key>
        """
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        auth = request.headers.get("Authorization")
        if not auth:
            return None
        # e.g. "Bearer abc123"
        parts = auth.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
        return None

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        
        # Skip authentication for health check, docs, and static files
        skip_paths = [
            "/v1/health",
            "/v1/healthz",
            "/api/v1/health",
            "/api/v1/healthz",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
        ]
        
        # Skip authentication for paths that don't start with protected prefixes
        path = request.url.path
        protected_prefixes = ("/v1/", "/api/v1/", "/rag/")
        if path in skip_paths or not path.startswith(protected_prefixes):
            return await call_next(request)
        
        # If no API keys configured, skip authentication
        if not settings.api_key and not settings.api_keys:
            return await call_next(request)
        
        # Get API key from header (X-API-Key or Authorization: Bearer)
        api_key = self._extract_api_key(request)
        
        if not api_key:
            return ORJSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "missing_api_key_header"},
            )
        
        # Validate API key
        valid_keys = []
        if settings.api_key:
            valid_keys.append(settings.api_key)
        if settings.api_keys:
            valid_keys.extend([k.strip() for k in settings.api_keys.split(",")])
        
        if api_key not in valid_keys:
            return ORJSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "invalid_api_key"},
            )
        
        return await call_next(request)

from __future__ import annotations

import redis
from redis import Redis

from app.settings import get_settings

_REDIS_CLIENT: Redis | None = None


def get_redis_client() -> Redis:
    """Get or create Redis client singleton."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        settings = get_settings()
        _REDIS_CLIENT = redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            decode_responses=True,  # Return strings instead of bytes
        )
    return _REDIS_CLIENT

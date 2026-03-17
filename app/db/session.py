from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from app.settings import get_settings

logger = logging.getLogger(__name__)


def _async_url(url: str) -> tuple[str, dict[str, Any]]:
    """
    Convert postgresql:// or postgres:// to postgresql+asyncpg://.

    Strips ALL query-string parameters (sslmode, channel_binding, etc.)
    because they are libpq/psycopg-specific and asyncpg rejects them.
    If sslmode was require/verify-*, we pass ssl=True via connect_args.
    """
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            rest = url[len(prefix):]
            parsed = urlparse("postgresql://" + rest)
            query = parse_qs(parsed.query, keep_blank_values=True)

            sslmode = (query.get("sslmode") or [None])[0]

            # Build clean URL with NO query params — asyncpg doesn't use any
            new_url = "postgresql+asyncpg://" + parsed.netloc + (parsed.path or "")

            connect_args: dict[str, Any] = {}
            if sslmode and sslmode.lower() in ("require", "verify-ca", "verify-full"):
                connect_args["ssl"] = True

            return new_url, connect_args
    return url, {}


_settings = get_settings()
_db_url, _db_connect_args = _async_url(_settings.database_url)

_use_null_pool = "neon.tech" in _db_url or _db_url.startswith("sqlite")

if _use_null_pool:
    engine = create_async_engine(
        _db_url, echo=False, poolclass=NullPool, connect_args=_db_connect_args
    )
else:
    _connect_args = dict(_db_connect_args)
    # asyncpg connection timeout (default 60s); avoid worker exit on slow DB
    _connect_args.setdefault("timeout", 60)
    engine = create_async_engine(
        _db_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_timeout=60,
        pool_recycle=1800,
        pool_pre_ping=True,
        connect_args=_connect_args,
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator for FastAPI ``Depends``."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside of FastAPI dependency injection
    (worker, tasks, internal helpers).

    Usage::

        async with get_session_ctx() as session:
            result = await session.execute(select(Model))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables and run idempotent migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    try:
        from app.db.migrations import run_all_migrations
        await asyncio.to_thread(run_all_migrations)
    except Exception:
        logger.warning("migrations_skipped_during_startup", exc_info=True)

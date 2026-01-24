from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from app.settings import get_settings


def get_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        return create_engine(settings.database_url, echo=False, connect_args=connect_args)

    # Postgres / other network DBs: avoid stale pooled connections causing
    # "server closed the connection unexpectedly" OperationalError.
    # Convert postgresql:// to postgresql+psycopg:// for psycopg3
    db_url = settings.database_url
    if db_url.startswith("postgresql://") and not db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    return create_engine(
        db_url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


ENGINE = None


def init_db() -> None:
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()
    SQLModel.metadata.create_all(ENGINE)


def get_session() -> Session:
    if ENGINE is None:
        init_db()
    return Session(ENGINE)



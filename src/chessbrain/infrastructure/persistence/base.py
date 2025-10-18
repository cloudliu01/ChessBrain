from __future__ import annotations

from typing import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from src.chessbrain.infrastructure.config import AppConfig, load_config

Base = declarative_base()


def create_engine_from_config(config: AppConfig | None = None):
    """Create a SQLAlchemy engine using the provided configuration."""
    cfg = config or load_config()
    engine_kwargs: dict = {"pool_pre_ping": True, "future": True}

    if cfg.database_url.startswith("sqlite"):
        engine_kwargs.update(
            {
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            }
        )

    return create_engine(cfg.database_url, **engine_kwargs)


def create_session_factory(config: AppConfig | None = None) -> sessionmaker:
    """Produce a session factory tied to the application engine."""
    engine = create_engine_from_config(config=config)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


SessionLocal = scoped_session(create_session_factory())


@contextmanager
def session_scope(config: AppConfig | None = None) -> Iterator:
    """Provide a transactional session scope."""
    session_factory = SessionLocal if config is None else scoped_session(
        create_session_factory(config=config)
    )
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "Base",
    "SessionLocal",
    "create_engine_from_config",
    "create_session_factory",
    "session_scope",
]

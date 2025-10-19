from __future__ import annotations

from pathlib import Path
import pytest
from sqlalchemy.orm import sessionmaker

from src.chessbrain.infrastructure.config import AppConfig
from src.chessbrain.infrastructure.persistence.base import (
    Base,
    create_engine_from_config,
)
from src.chessbrain.infrastructure.persistence import game_session_repository  # noqa: F401
from src.chessbrain.interface.http.app import create_app


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Provide a configuration tuned for isolated tests."""
    return AppConfig(
        database_url="sqlite+pysqlite:///:memory:",
        model_checkpoint_dir=Path("tests/.artifacts/checkpoints").resolve(),
        tensorboard_log_dir=Path("tests/.artifacts/tensorboard").resolve(),
        flask_env="test",
        pytorch_device_preference="cpu",
        additional={},
    )


@pytest.fixture(scope="session")
def engine(app_config: AppConfig):
    engine = create_engine_from_config(app_config)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(engine):
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app(app_config: AppConfig):
    flask_app = create_app(app_config)
    flask_app.config.update(TESTING=True)
    return flask_app

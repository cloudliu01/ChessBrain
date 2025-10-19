from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Centralized runtime configuration for backend services."""

    database_url: str
    model_checkpoint_dir: Path
    tensorboard_log_dir: Path
    flask_env: str = "production"
    pytorch_device_preference: str = "auto"
    additional: dict[str, str] = field(default_factory=dict)


def load_config(prefix: str = "") -> AppConfig:
    """Load application configuration from environment variables."""

    def _get_env(key: str, default: str = "") -> str:
        env_key = f"{prefix}{key}"
        return os.getenv(env_key, default)

    database_url = _get_env("DATABASE_URL", "postgresql+psycopg://localhost/chessbrain")
    checkpoint_dir = Path(_get_env("MODEL_CHECKPOINT_DIR", "models/checkpoints")).resolve()
    tensorboard_dir = Path(_get_env("TENSORBOARD_LOG_DIR", "data/tensorboard")).resolve()

    additional_keys = (
        "STRUCTLOG_LEVEL",
        "JWT_SECRET",
        "TRAINING_JOBS_ROOT",
    )
    additional: dict[str, str] = {}
    for key in additional_keys:
        value = _get_env(key, "")
        if value:
            additional[key] = value

    return AppConfig(
        database_url=database_url,
        model_checkpoint_dir=checkpoint_dir,
        tensorboard_log_dir=tensorboard_dir,
        flask_env=_get_env("FLASK_ENV", "production"),
        pytorch_device_preference=_get_env("PYTORCH_DEVICE", "auto"),
        additional=additional,
    )


__all__ = ["AppConfig", "load_config"]

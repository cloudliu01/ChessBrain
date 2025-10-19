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
    training_learning_rate: float = 1e-3
    training_l2_coefficient: float = 1e-4
    active_model_path: Path | None = None
    active_model_version: str | None = None


def load_config(prefix: str = "") -> AppConfig:
    """Load application configuration from environment variables."""

    def _get_env(key: str, default: str = "") -> str:
        env_key = f"{prefix}{key}"
        return os.getenv(env_key, default)

    database_url = _get_env("DATABASE_URL", "postgresql+psycopg://localhost/chessbrain")
    checkpoint_dir = Path(_get_env("MODEL_CHECKPOINT_DIR", "models/checkpoints")).resolve()
    tensorboard_dir = Path(_get_env("TENSORBOARD_LOG_DIR", "data/tensorboard")).resolve()
    active_model_raw = _get_env("ACTIVE_MODEL_PATH", "")
    active_model_path = Path(active_model_raw).resolve() if active_model_raw else None
    active_model_version = _get_env("ACTIVE_MODEL_VERSION", "") or None
    if not active_model_version and active_model_path is not None:
        active_model_version = active_model_path.stem

    learning_rate_raw = _get_env("TRAINING_LEARNING_RATE", "0.001")
    l2_raw = _get_env("TRAINING_L2_COEFFICIENT", "0.0001")

    def _parse_float(raw: str, fallback: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    training_learning_rate = _parse_float(learning_rate_raw, 0.001)
    training_l2_coefficient = _parse_float(l2_raw, 0.0001)

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
        active_model_path=active_model_path,
        active_model_version=active_model_version,
        training_learning_rate=training_learning_rate,
        training_l2_coefficient=training_l2_coefficient,
    )


__all__ = ["AppConfig", "load_config"]

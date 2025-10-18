from __future__ import annotations

from flask import Flask

from src.chessbrain.infrastructure.config import AppConfig, load_config
from src.chessbrain.interface.http.model_routes import model_bp
from src.chessbrain.interface.telemetry.logging import setup_logging, get_logger


def create_app(config: AppConfig | None = None) -> Flask:
    """Instantiate Flask application with shared configuration."""
    cfg = config or load_config()

    setup_logging(cfg.additional.get("STRUCTLOG_LEVEL", "INFO"))
    logger = get_logger("chessbrain.app")

    app = Flask(__name__)
    app.config.update(
        DATABASE_URL=cfg.database_url,
        MODEL_CHECKPOINT_DIR=str(cfg.model_checkpoint_dir),
        TENSORBOARD_LOG_DIR=str(cfg.tensorboard_log_dir),
        PYTORCH_DEVICE_PREF=cfg.pytorch_device_preference,
        ENV=cfg.flask_env,
    )

    app.register_blueprint(model_bp, url_prefix="/api/v1/models")

    @app.get("/healthz")
    def healthcheck():
        return {"status": "ok"}, 200

    logger.info(
        "flask_app_initialized",
        env=cfg.flask_env,
        model_checkpoint_dir=str(cfg.model_checkpoint_dir),
    )
    return app


__all__ = ["create_app"]

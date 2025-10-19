from __future__ import annotations

from flask import Flask
from sqlalchemy.orm import sessionmaker

from src.chessbrain.infrastructure.config import AppConfig, load_config
from src.chessbrain.infrastructure.persistence.base import Base, create_engine_from_config
from src.chessbrain.interface.http.gameplay_routes import gameplay_bp
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
        APP_CONFIG=cfg,
    )
    if cfg.active_model_path is not None:
        app.config["ACTIVE_MODEL_PATH"] = str(cfg.active_model_path)
    if cfg.active_model_version is not None:
        app.config["ACTIVE_MODEL_VERSION"] = cfg.active_model_version

    engine = create_engine_from_config(cfg)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    app.config["SESSION_FACTORY"] = session_factory

    app.register_blueprint(model_bp, url_prefix="/api/v1/models")
    app.register_blueprint(gameplay_bp, url_prefix="/api/v1/sessions")

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

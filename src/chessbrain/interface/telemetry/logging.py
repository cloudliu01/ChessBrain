from __future__ import annotations

from typing import Any
import logging
import sys
import structlog


def setup_logging(level: int | str = "INFO") -> None:
    """Configure structlog for JSON-formatted logging with trace IDs."""
    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
    )

    if isinstance(level, str):
        min_level = logging.getLevelName(level.upper())
        if not isinstance(min_level, int):
            min_level = logging.INFO
    else:
        min_level = level

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            min_level
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    """Return a structlog logger bound to the provided name."""
    return structlog.get_logger(name or "chessbrain")


def bind_trace(logger: Any, trace_id: str | None = None, **kwargs) -> Any:
    """Attach trace metadata to a logger for request correlation."""
    context = {"trace_id": trace_id} if trace_id else {}
    context.update(kwargs)
    return logger.bind(**context)


__all__ = ["setup_logging", "get_logger", "bind_trace"]

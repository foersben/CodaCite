"""Centralized logging configuration for the application."""

import logging
import logging.config
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="system")


class RequestIdFilter(logging.Filter):
    """Logging filter to inject request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject the current request_id into the log record."""
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> None:
    """Initialize standard logging configuration."""
    log_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": "app.core.logging_config.RequestIdFilter",
            }
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [req_id: %(request_id)s] - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["request_id"],
                "level": "INFO",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
                "formatter": "standard",
                "filters": ["request_id"],
                "level": "DEBUG",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
    }
    logging.config.dictConfig(log_config)

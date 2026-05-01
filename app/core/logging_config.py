"""Centralized logging configuration for the application.

This module provides the logging setup for the entire application, including
request ID filtering and rotating file handlers.
"""

import logging
import logging.config
from contextvars import ContextVar

from app.config import settings

# Context variable to hold the current request ID for logging
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="system")


class RequestIdFilter(logging.Filter):
    """Logging filter to inject request_id into log records.

    This filter ensures that the current request ID from the context variable
    is available for the log formatter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject the current request_id into the log record.

        Args:
            record: The log record to be filtered/modified.

        Returns:
            Always returns True to indicate the record should be processed.
        """
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> None:
    """Initialize standard logging configuration.

    Sets up the logging dictionary configuration with filters, formatters,
    and handlers for console and file output.
    """
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
                "filename": str(settings.logs_dir / "app.log"),
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

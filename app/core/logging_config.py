"""Centralized logging configuration for the application.

This module provides the logging setup for the entire application, including
request ID filtering and rotating file handlers.
"""

import logging
import logging.config
import os
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
    handlers = ["console"]
    file_handler_config = None

    # Attempt to prepare the file handler
    try:
        log_file = settings.logs_dir / "app.log"
        # Ensure directory exists
        settings.logs_dir.mkdir(parents=True, exist_ok=True)

        # Verify we can write to the directory by touching the file or checking permissions
        if os.access(settings.logs_dir, os.W_OK):
            handlers.append("file")
            file_handler_config = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
                "formatter": "standard",
                "filters": ["request_id"],
                "level": "DEBUG",
            }
    except Exception as e:
        # We use print here because logging isn't set up yet
        print(f"WARNING: Could not initialize file logging: {e}. Falling back to console only.")

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
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "[req_id: %(request_id)s] - %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["request_id"],
                "level": "INFO",
            },
        },
        "root": {
            "handlers": handlers,
            "level": "DEBUG",
        },
    }

    if file_handler_config:
        log_config["handlers"]["file"] = file_handler_config  # type: ignore

    logging.config.dictConfig(log_config)

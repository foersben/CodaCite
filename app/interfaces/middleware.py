"""FastAPI middleware for request lifecycle and logging.

This module provides middleware to track request context and log performance
metrics for every incoming HTTP request.
"""

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging_config import request_id_ctx

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to assign UUID to requests and log duration.

    This middleware injects a unique request ID into the logging context and
    logs the start, end, and duration of every HTTP request.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request, setting context and logging start/end.

        Args:
            request: The incoming FastAPI request object.
            call_next: The next handler in the middleware chain.

        Returns:
            The FastAPI response object.

        Raises:
            Exception: Re-raises any exceptions encountered during request processing.
        """
        req_id = str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        start_time = time.time()

        logger.info("Started %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                "Completed %s %s | Status: %s | Time: %.4fs",
                request.method,
                request.url.path,
                response.status_code,
                process_time,
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Failed %s %s | Time: %.4fs | Error: %s",
                request.method,
                request.url.path,
                process_time,
                str(e),
            )
            raise
        finally:
            request_id_ctx.reset(token)

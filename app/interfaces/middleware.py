"""FastAPI middleware for request lifecycle and logging."""

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging_config import request_id_ctx

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to assign UUID to requests and log duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request, setting context and logging start/end."""
        req_id = str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        start_time = time.time()

        logger.info(f"Started processing request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"Completed processing request: {request.method} {request.url.path} "
                f"with status code {response.status_code} in {process_time:.4f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {process_time:.4f}s. Error: {e}"
            )
            raise
        finally:
            request_id_ctx.reset(token)

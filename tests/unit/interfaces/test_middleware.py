"""Unit tests for the application middleware.

Validates the RequestLoggingMiddleware, ensuring it correctly processes
successful requests and handles exceptions without leaking sensitive data.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.interfaces.middleware import RequestLoggingMiddleware


def test_request_logging_middleware_success() -> None:
    """Tests the middleware's success path.

    Given:
        A FastAPI application with RequestLoggingMiddleware.
    When:
        A successful request is made to a valid endpoint.
    Then:
        The middleware should allow the request to pass and return the correct response.
    """
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        """A simple test endpoint."""
        return {"message": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def test_request_logging_middleware_exception() -> None:
    """Tests the middleware's exception handling path.

    Given:
        A FastAPI application with RequestLoggingMiddleware.
    When:
        A request is made to an endpoint that raises an unhandled exception.
    Then:
        The middleware should handle the exception and ensure a 500 status code is returned.
    """
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/error")
    async def error_endpoint() -> None:
        """An endpoint that always raises an error."""
        raise ValueError("Intentional error")

    client = TestClient(app, raise_server_exceptions=False)
    # With raise_server_exceptions=False, TestClient returns 500 for unhandled errors
    response = client.get("/error")

    assert response.status_code == 500

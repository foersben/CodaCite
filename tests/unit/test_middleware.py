"""Tests for the application middleware.

This module validates the RequestLoggingMiddleware in app/interfaces/middleware.py.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.interfaces.middleware import RequestLoggingMiddleware


def test_request_logging_middleware_success():
    """Test the middleware success path."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        """Docstring generated to satisfy ruff D103."""
        return {"message": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def test_request_logging_middleware_exception():
    """Test the middleware exception path."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/error")
    async def error_endpoint():
        """Docstring generated to satisfy ruff D103."""
        raise ValueError("Intentional error")

    client = TestClient(app, raise_server_exceptions=False)
    # With raise_server_exceptions=False, TestClient returns 500
    response = client.get("/error")

    assert response.status_code == 500

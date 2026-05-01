"""Main entry point for the FastAPI application.

This module initializes the FastAPI app, configures middleware, includes routers,
and sets up lifecycle event handlers for database initialization.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_resource_path
from app.core.logging_config import setup_logging
from app.infrastructure.bootstrap import ensure_models_exist
from app.interfaces.dependencies import init_db
from app.interfaces.middleware import RequestLoggingMiddleware
from app.interfaces.routers import api_router

# Initialize centralized logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    logger.info("Starting up CodaCite")
    await init_db()
    yield
    logger.info("Shutting down CodaCite")


app = FastAPI(
    title="CodaCite",
    description="GraphRAG-based Document Intelligence with verifiable citations",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# Configure global middleware
app.add_middleware(RequestLoggingMiddleware)

# Include API endpoints
app.include_router(api_router)

# Mount static files using resource path helper
app.mount("/static", StaticFiles(directory=str(get_resource_path("app/static"))), name="static")

# Set up templates using resource path helper
templates = Jinja2Templates(directory=str(get_resource_path("app/templates")))


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """Serve the NotebookLM-like UI at the root.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The rendered notebook.html template.
    """
    return templates.TemplateResponse(request=request, name="notebook.html")


if __name__ == "__main__":
    import asyncio

    import uvicorn

    # 1. Run the bootstrap process to ensure models are present
    print("\n--- CodaCite Engine Initialization ---")
    try:
        asyncio.run(ensure_models_exist())
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during bootstrap: {e}")
        import sys

        sys.exit(1)

    # 2. Start the FastAPI server
    print("\n🚀 Starting CodaCite Local Server...")
    print("UI available at: http://localhost:8080")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, log_level="warning")

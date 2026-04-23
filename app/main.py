"""Main entry point for the FastAPI application.

This module initializes the FastAPI app, configures middleware, includes routers,
and sets up lifecycle event handlers for database initialization.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.logging_config import setup_logging
from app.interfaces.dependencies import init_db
from app.interfaces.middleware import RequestLoggingMiddleware
from app.interfaces.routers import api_router

# Initialize centralized logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Omni-Copilot",
    description="GraphRAG-based Document Intelligence and Workflow Automation",
    version="0.1.0",
)

# Configure global middleware
app.add_middleware(RequestLoggingMiddleware)

# Include API endpoints
app.include_router(api_router)

# Set up templates for the frontend UI
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """Serve the NotebookLM-like UI at the root.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The rendered notebook.html template.
    """
    return templates.TemplateResponse(request=request, name="notebook.html")


@app.on_event("startup")
async def startup_event() -> None:
    """Run upon application startup.

    Initializes the database connection and ensures the schema is ready.
    """
    logger.info("Starting up Enterprise Omni-Copilot")
    await init_db()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run upon application shutdown.

    Performs necessary cleanup tasks.
    """
    logger.info("Shutting down Enterprise Omni-Copilot")


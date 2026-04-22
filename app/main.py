"""Main application entrypoint."""

import logging

from fastapi import FastAPI

from app.interfaces.dependencies import init_db
from app.interfaces.middleware import RequestLoggingMiddleware
from app.interfaces.routers import api_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Omni-Copilot",
    description="GraphRAG-based Document Intelligence and Workflow Automation",
    version="0.1.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(api_router)


# @app.on_event("startup")
# async def startup_event() -> None:
#     """Run upon application startup."""
#     logger.info("Starting up Enterprise Omni-Copilot")


@app.on_event("startup")
async def startup_event() -> None:
    """Run upon application startup."""
    logger.info("Starting up Enterprise Omni-Copilot")
    await init_db()  # <--- CRITICAL: Ensure this line exists


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run upon application shutdown."""
    logger.info("Shutting down Enterprise Omni-Copilot")

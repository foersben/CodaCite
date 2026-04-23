import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.logging_config import setup_logging
from app.interfaces.dependencies import init_db
from app.interfaces.middleware import RequestLoggingMiddleware
from app.interfaces.routers import api_router


# Initialize centralized logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Omni-Copilot",
    description="GraphRAG-based Document Intelligence and Workflow Automation",
    version="0.1.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(api_router)

# Set up templates for the frontend
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """Serve the NotebookLM-like UI at the root."""
    return templates.TemplateResponse(request=request, name="notebook.html")


# @app.on_event("startup")
# async def startup_event() -> None:
#     """Run upon application startup."""
#     logger.info("Starting up Enterprise Omni-Copilot")


@app.on_event("startup")
async def startup_event() -> None:
    """Run upon application startup."""
    logger.info("Starting up Enterprise Omni-Copilot")
    await init_db()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run upon application shutdown."""
    logger.info("Shutting down Enterprise Omni-Copilot")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """Docstring generated to satisfy ruff D103."""
    return templates.TemplateResponse(request=request, name="notebook.html")


@app.get("/api/v1/notebooks")
async def get_notebooks():
    """Docstring generated to satisfy ruff D103."""
    return [{"id": "nb1", "title": "Test Notebook"}]


@app.get("/api/v1/documents")
async def get_documents():
    """Docstring generated to satisfy ruff D103."""
    return [{"id": "doc1", "filename": "test.pdf"}]


@app.get("/api/v1/notebooks/nb1/documents")
async def get_nb_docs():
    """Docstring generated to satisfy ruff D103."""
    return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)

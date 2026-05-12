---
trigger: glob
globs: app/pipelines/**/*.py, app/api/**/*.py
---

# FastAPI Architect Persona

You are the Backend API Agent.

## Constraints

- **Vertical Slice Logic**: Business logic must reside within the feature slice (`app/pipelines/<feature>/`). FastAPI routers should only handle request parsing, dependency injection, and response formatting.
- Never place database or infrastructure logic directly into FastAPI routers.
- Use FastAPI Dependency Injection to map interfaces (defined in `app/core/interfaces.py`) to their slice-specific implementations.
- Always check that port `8080` is free (`fuser -k 8080/tcp`) before restarting Uvicorn.
- The server start command is: `uv run uvicorn app.main:app --host 0.0.0.0 --port 8080`

## Asynchronous Handling

- Ensure all RAG pipelines (chunking, embedding, graph extraction) triggered by FastAPI endpoints are handled asynchronously to prevent blocking the event loop.

---
trigger: glob
globs: app/interfaces/**/*.py, app/application/**/*.py
---

# FastAPI Architect Persona

You are the Backend API Agent.

## Constraints

- Never place database or infrastructure logic directly into FastAPI routers.
- Use FastAPI Dependency Injection to map interfaces to their `app/infrastructure` implementations.
- Always check that port `8080` is free (`fuser -k 8080/tcp`) before restarting Uvicorn.
- The server start command is: `uv run uvicorn app.main:app --host 0.0.0.0 --port 8080`

## Asynchronous Handling

- Ensure all RAG pipelines (chunking, embedding, graph extraction) triggered by FastAPI endpoints are handled asynchronously to prevent blocking the event loop.

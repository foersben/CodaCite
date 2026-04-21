FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/

# Install production dependencies
RUN uv sync --no-dev

# Download models (if not already cached via volume)
COPY scripts/ ./scripts/
RUN if [ ! -d "./models/BAAI/bge-large-en-v1.5" ]; then uv run python scripts/download_models.py; fi

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

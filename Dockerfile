FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/
COPY scripts/ ./scripts/

# Install production dependencies only (models are volume-mounted at runtime)
RUN uv sync --no-dev

EXPOSE 8080

# Models are expected at /app/models via a volume mount.
# To pre-download them into the volume: docker run --rm -v models:/app/models <image> uv run python scripts/download_models.py
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

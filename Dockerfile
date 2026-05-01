FROM python:3.13-slim

# Set environment variables for non-interactive apt-get and Python
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# uv environment variables
ENV UV_CACHE_DIR=/app/.uv_cache
ENV UV_PYTHON_INSTALL_DIR=/app/.uv_python
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install necessary system dependencies for building certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone installer
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Copy project files
COPY pyproject.toml uv.lock README.md ./
# We copy the code, but for dev we will often mount it anyway
COPY app/ ./app/

# Install production dependencies
# Note: In production builds, we run this. In local dev, we will override
# the .venv with a volume mount from the host.
RUN uv sync --no-dev

EXPOSE 8080

# Run the app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

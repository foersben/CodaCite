FROM python:3.11-slim

# Set environment variables for non-interactive apt-get and Python
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install necessary system dependencies for building certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone installer
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Copy project files
COPY pyproject.toml README.md ./
COPY app/ ./app/

# Install production dependencies only
RUN uv sync --no-dev

EXPOSE 8080

# Run the app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

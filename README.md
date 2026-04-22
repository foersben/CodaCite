# Enterprise Omni-Copilot

GraphRAG-based Document Intelligence and Workflow Automation system.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Run](#run)
- [Usage](#usage)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [References](#references)

## Overview

Enterprise Omni-Copilot is a cutting-edge system designed to streamline document intelligence and workflow automation. Built on the GraphRAG framework, it integrates advanced natural language processing (NLP) models and graph-based reasoning to:

- Extract insights from unstructured data.
- Automate complex workflows.
- Enable seamless integration with enterprise systems.

Key features include:

- **FastAPI-based API** for scalable and efficient interactions.
- **Pre-trained NLP models** for coreference resolution, entity extraction, and more.
- **Graph-based reasoning** for advanced data relationships.
- **Extensible architecture** for custom workflows.

## Setup

### Local Setup

```bash
uv sync
uv run download-models
```

### Docker Setup

To build and run the application using Docker:

```bash
docker build -t enterprise-omni-copilot .
docker run -p 8080:8080 enterprise-omni-copilot
```

## Run

### Start the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The API is then available at `http://localhost:8080`.

### API Documentation

Open `http://localhost:8080/docs` for the interactive Swagger UI.

## Usage

### CLI Commands

The project provides a CLI for managing models and running tasks. Key commands include:

- **Download Models**:

```bash
uv run download-models
```

Downloads the required NLP models to the `models/` directory.

### API Usage

The application exposes a FastAPI-based REST API. Once the server is running, you can interact with the API:

- **Swagger UI**:

  Visit `http://localhost:8080/docs` to explore the API documentation and test endpoints interactively.

- **Example Request**:

```bash
curl -X POST "http://localhost:8080/api/v1/process" -H "Content-Type: application/json" -d '{"text": "Your input text here"}'
```

### Ingest Documents

Use the ingest endpoint to parse a document, chunk it, and update the GraphRAG state.

- **Supported formats**: `.pdf`, `.docx`, `.md`, `.markdown`, `.txt`
- **Endpoint**: `POST /api/v1/ingest` with `multipart/form-data` field `file`

Example:

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -H "accept: application/json" \
  -F "file=@sample.pdf;type=application/pdf"
```

Expected successful response:

```json
{
  "filename": "sample.pdf",
  "chunks_processed": 12,
  "entities_extracted": 34
}
```

Validation behavior:

- Unsupported file extensions return `400 Bad Request`.
- Corrupted or unreadable documents return `400 Bad Request` with a parse error detail.

Troubleshooting PDF ingest:

- Ensure the PDF is not encrypted or corrupted.
- Ensure the PDF contains selectable text (image-only scans may extract little or no text).
- Verify the file extension matches the actual file format.

### Dockerized Deployment

To run the application in a Docker container:

1. Build the Docker image:

   ```bash
   docker build -t enterprise-omni-copilot .
   ```

2. Run the container:

   ```bash
   docker run -p 8080:8080 enterprise-omni-copilot
   ```

3. Access the API at `http://localhost:8080`.

## Architecture

### High-Level Overview

The system is designed with modularity and scalability in mind. Key components include:

- **Core**:
Centralized logging configuration and shared utilities.

- **Domain**:
Business logic, including models, exceptions, and ports.

- **Infrastructure**:
Integrations with external systems (e.g., NLP models, databases).

- **Interfaces**:
API layer, middleware, and dependency injection.

### Key Modules

- **`app/core`**:
Handles logging and other core configurations.

- **`app/interfaces`**:
Defines the FastAPI routers and middleware.

- **`app/infrastructure`**:
Implements NLP model loading and database interactions.

- **`app/domain`**:
Contains the domain models and business logic.

### Data Flow

1. **Ingestion**:
Data is received via API endpoints.

2. **Processing**:
NLP models process the data (e.g., coreference resolution, entity extraction).

3. **Storage**:
Results are stored in the database or returned to the client.

## Contributing

Contributions are welcome. Please run checks before opening a PR:

```bash
uv run ruff check app tests
uv run mypy app
uv run pytest
```

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [GraphRAG](https://github.com/GraphRAG)

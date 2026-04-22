# Enterprise Omni-Copilot

GraphRAG-based Document Intelligence and Workflow Automation system.

## Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview)
  - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
  - [Setup](https://www.google.com/search?q=%23setup)
  - [Run](https://www.google.com/search?q=%23run)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Architecture](https://www.google.com/search?q=%23architecture)
  - [Contributing](https://www.google.com/search?q=%23contributing)

## Overview

Enterprise Omni-Copilot is a system designed to streamline document intelligence and workflow automation using the GraphRAG framework. It integrates advanced NLP models and graph-based reasoning to:

  - Extract insights from unstructured data.
  - Automate complex workflows.
  - Enable seamless integration with enterprise systems.

**Key features include:**

  - **FastAPI-based API** for scalable interactions.
  - **Pre-trained NLP models** for coreference resolution and entity extraction.
  - **Graph-based reasoning** for advanced data relationships.
  - **Extensible architecture** following hexagonal patterns (Domain, Infrastructure, Interfaces).

## Prerequisites

### 1\. Database (SurrealDB)

This application requires **SurrealDB** as its graph and document store. You must have an instance running before starting the app.

**Start with Docker:**

```bash
docker run --rm -p 8000:8000 surrealdb/surrealdb:latest start --user root --pass root memory
```

*Note: The application connects to `ws://localhost:8000` by default.*

### 2\. Environment Variables

The system uses Google Gemini for structured graph extraction. Set your API key using your local secret tool:

```bash
export GEMINI_API_KEY=$(secret-tool lookup Title Gemini_API)
```

## Setup

### Local Setup

```bash
# Install dependencies
uv sync

# Download required NLP model artifacts (BAAI/bge-large-en-v1.5)
uv run download-models
```

### Docker Setup

To build and run the application stack using Docker Compose:

```bash
docker-compose up --build
```

## Run

### Start the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The API is available at `http://localhost:8080`. Note that the root path (`/`) will return a 404; use the documentation link below to verify the server is up.

### API Documentation

Open [http://localhost:8080/docs](https://www.google.com/search?q=http://localhost:8080/docs) for the interactive Swagger UI.

## Usage

### Ingest Documents

Use the ingest endpoint to parse a document, resolve coreferences, and update the GraphRAG state.

  - **Supported formats**: `.pdf`, `.docx`, `.md`, `.markdown`, `.txt`
  - **Endpoint**: `POST /api/v1/ingest`

<!-- end list -->

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -H "accept: application/json" \
  -F "file=@sample.pdf"
```

### Query the Graph

Query the knowledge base using hybrid retrieval (vector search + graph traversal).

  - **Endpoint**: `POST /api/v1/query`

<!-- end list -->

```bash
curl -X POST "http://localhost:8080/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the core findings?", "top_k": 5}'
```

## Architecture

### Key Modules

  - **`app/core`**: Centralized logging and shared utilities.
  - **`app/domain`**: Business logic, including Pydantic models (Node, Edge, Chunk) and abstract ports.
  - **`app/infrastructure`**: Concrete implementations for SurrealDB, Gemini Extraction, and Jaro-Winkler Entity Resolution.
  - **`app/interfaces`**: FastAPI routers, dependency injection, and request logging middleware.

### Data Flow

1.  **Ingestion**: Data is received, coreferences are resolved, and text is chunked.
2.  **Processing**: Entities and relationships are extracted via LLM or local fallback.
3.  **Storage**: Data is persisted in SurrealDB with HNSW vector indices.

## Contributing

Please run checks before opening a PR:

```bash
uv run ruff check app tests
uv run mypy app
uv run pytest
```
# CodaCite

GraphRAG-based Document Intelligence with a premium, NotebookLM-inspired interface.

**CodaCite** stands for **C**ontextual **O**mni-**D**ocument **A**ssistant with **Cite**-ability. It is designed to provide verifiable, grounded intelligence from large document collections.

## Table of Contents

- [Overview](#overview)
- [New: Notebook UI](#notebook-ui)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run](#run)
- [Usage](#usage)
- [Architecture](#architecture)
- [Contributing](#contributing)

## Overview

**CodaCite** is a system designed to streamline document intelligence and workflow automation using the GraphRAG framework. It integrates advanced NLP models and graph-based reasoning to:

- Extract insights from unstructured data.

- Automate complex workflows.

- Enable seamless interaction via a high-impact, modern web interface.

**Key features include:**

- **NotebookLM-style UI**: A premium, dark-mode interface for document management and contextual chat.

- **FastAPI Backend**: Scalable API following Hexagonal Architecture.

- **Hybrid Retrieval**: Combines vector search with graph traversal for deep contextual reasoning.
- **Drag-and-Drop Ingestion**: Upload documents directly via the browser for automated graph extraction.

## Notebook UI

The application now features a state-of-the-art "Notebook" interface at the root path (`/`).

### Key Features

- **Sidebar Document Manager**: View all ingested files and their metadata.
- **Interactive Chat**: Multi-turn conversation grounded in your private document graph.
- **Markdown Rendering**: Beautifully formatted responses with support for code blocks, lists, and LaTeX-style math.
- **Copy as Markdown**: Easily retrieve raw markdown content from assistant responses for use in other tools.
- **Fluid Responsiveness**: A "large-format" design (1.5x scale) optimized for high readability and professional use.

## Prerequisites

### 1. Database (SurrealDB)

This application requires **SurrealDB** as its graph and document store.

**Start with Docker:**

```bash
docker run --rm -p 8000:8000 surrealdb/surrealdb:v1.5.4 start --user root --pass root memory
```

*Note: The application connects to `ws://localhost:8000` by default.*

## Environment Variables

The system uses Google Gemini for structured graph extraction and chat generation. You can provide the API key in two ways:

### 1. Secret Service (Recommended)

If you use **KeePassXC** (or another Secret Service compatible manager), the application can retrieve the key automatically.

- **Service Name (Title)**: `Gemini_API`
- **Account Name (Username)**: `gemini_user`

The application will attempt to fetch this key if the `GEMINI_API_KEY` environment variable is not set.

### 2. Manual Export

Alternatively, you can set the key manually in your shell or `.env` file:

```bash
export GEMINI_API_KEY="your-api-key-here"
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

- **Main UI**: [http://localhost:8080/](http://localhost:8080/)
- **API Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

## Usage

### Ingesting Documents

You can ingest documents in two ways:

1. **Web UI**: Drag and drop files into the sidebar "Drop Zone".
2. **API**: Send a `POST` request to `/api/v1/ingest`.

**Supported formats**: `.pdf`, `.docx`, `.md`, `.markdown`, `.txt`

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -H "accept: application/json" \
  -F "file=@sample.pdf"
```

### Chatting with Documents

Use the Web UI for the best experience, or use the chat endpoint:

- **Endpoint**: `POST /api/v1/chat`
- **Payload**: `{"message": "What is this document about?", "history": []}`

## Architecture

Following **Hexagonal Architecture** for maximum maintainability:

- **`app/domain`**: Pure logic and Pydantic models (Node, Edge, Chunk). No external dependencies.
- **`app/infrastructure`**: Concrete implementations for SurrealDB, Gemini API, and Embeddings.
- **`app/application`**: Use cases coordinating domain logic and infrastructure.
- **`app/interfaces`**: FastAPI routers, templates, and middleware.

## Contributing

Please run checks before opening a PR:

```bash
uv run ruff check app tests
uv run mypy app
uv run pytest
```

### Documentation

Built with Zensical. To serve locally:

```bash
uv run zensical build
uv run zensical serve
```

# CodaCite

GraphRAG-based Document Intelligence with a premium, NotebookLM-inspired interface.

**CodaCite** stands for **C**ontextual **O**mni-**D**ocument **A**ssistant with **Cite**-ability. It is designed to provide verifiable, grounded intelligence from large document collections by bridging vector search with graph-based reasoning.

## Table of Contents

- [Overview](#overview)
- [Multi-Notebook Intelligence](#multi-notebook-intelligence)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run](#run)
- [Agent Workspace](#agent-workspace)
- [Architecture](#architecture)
- [Contributing](#contributing)

## Overview

**CodaCite** is a high-performance document intelligence engine that transforms unstructured data into an interconnected knowledge graph. It leverages the GraphRAG framework to enable deep semantic reasoning and automated workflow assistance.

**Key features include:**

- **Multi-Notebook Containers**: Partition knowledge into discrete containers. Scope retrieval to specific notebooks to maintain context isolation.
- **NotebookLM-inspired UI**: A premium, glassmorphic dark-mode interface for document management, multi-notebook selection, and contextual chat.
- **8-Phase Ingestion Pipeline**: Asynchronous pipeline handling coreference resolution (**FastCoref**), recursive chunking (**LangChain**), and LLM-based relationship extraction using **Google Gemini**.
- **Hybrid Retrieval**: Combines **SurrealDB HNSW** vector search with multi-hop graph traversals and **Cross-Encoder** reranking for panoramic context synthesis.

## Technology Stack & Concepts

CodaCite is built on a modern AI stack, prioritizing local performance and verifiable grounding.

### Core Libraries

- **Orchestration**: Custom Hexagonal implementation (inspired by LangChain/LangGraph patterns for modularity).
- **Text Processing**: `langchain-text-splitters` for `RecursiveCharacterTextSplitter`.
- **Linguistic Analysis**: `fastcoref` for high-performance coreference resolution.
- **Graph Algorithms**: `networkx` for Louvain community detection and graph synthesis.
- **Database**: **SurrealDB** (Multi-model: Document, Vector, and Graph).
- **ML Inference**: `OpenVINO` / `HuggingFace` for local embeddings (BGE-M3).
- **VCS & Env**: `uv` for package management, `Podman` for containerization.

### RAG Concepts

- **GraphRAG**: Bridging unstructured text chunks with structured knowledge graphs to enable hierarchical reasoning.
- **Hybrid Retrieval**: Fusing Vector Similarity (HNSW) with Graph Neighborhood Traversal (Multi-hop).
- **Community Detection**: Global graph summarization via cluster identification (Louvain).
- **Entity Resolution**: Deduplicating conceptual nodes using string similarity (Jaro-Winkler) and vector proximity.
- **Scoped Context**: Multi-notebook partitioning for logical data isolation and targeted retrieval.

## Multi-Notebook Intelligence

The application allows users to partition their knowledge base into "Notebooks." Rather than operating on a single monolithic store, you can create specific notebooks for different projects or domains.

- **Dynamic Scoping**: Select or deselect notebooks in the UI to instantly refine the AI's "active memory."
- **Graph-Based Isolation**: Documents are linked to notebooks via `belongs_to` relationships, ensuring that search results are strictly grounded in the user's selected context.
- **Responsive Management**: Create, rename, and populate notebooks with a drag-and-drop workflow.

## Prerequisites

### 1. Database (SurrealDB)

This application requires **SurrealDB** (v1.5+) as its graph and document store.

**Start with Podman:**

```bash
podman run --rm -p 8000:8000 surrealdb/surrealdb:v1.5.4 start --user root --pass root memory
```

*Note: The application connects to `ws://localhost:8000` by default. For persistent storage, use a local volume.*

### 2. Package Manager

The project strictly mandates the use of **uv**. Do not use `pip` or standard `venv`.

## Environment Variables

The system uses Google Gemini for structured graph extraction and chat generation. You can provide the API key in two ways:

### 1. Secret Service (Recommended)

If you use **KeePassXC** (or another Secret Service compatible manager), the application can retrieve the key automatically via D-Bus.

- **Service Name (Title)**: `Gemini_API`

### 2. Manual Export

Alternatively, you can set the key manually in your shell or `.env` file:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Setup

### Local Setup

Ensure the following environment variables are set to maintain an isolated, project-local development environment:

```bash
export UV_CACHE_DIR=$(pwd)/.uv_cache
export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv_python
```

```bash
# Install dependencies into .venv
uv sync

# Download required local NLP model artifacts (BAAI/bge-large-en-v1.5)
uv run download-models
```

### Podman Setup

To build and run the application stack using Podman Compose:

```bash
podman-compose up --build
```

## Run

### Start the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

- **Main UI**: [http://localhost:8080/](http://localhost:8080/)
- **API Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

## Agent Workspace

CodaCite is designed for agentic development. The following workflows are available within the agent interface:

| Command | Description |
| :--- | :--- |
| `/run_tests` | Runs ruff, mypy, and the full pytest suite. |
| `/commit` | Verifies code quality and handles the commit/push workflow. |
| `/implement` | Implements new features with automated planning and testing. |
| `/qa-pass` | Generates and verifies unit tests for a specific target file. |
| `/coverage-boost` | Audits the repository and writes tests to reach 90%+ coverage. |
| `/document-all` | Synchronizes docstrings and architectural documentation. |
| `/sync-zensical` | Updates the Zensical project documentation manifest. |
| `/update-readme` | Synchronizes the root README.md with the latest code. |

## Architecture

CodaCite follows a strict **Hexagonal Architecture** to isolate core logic from external infrastructure:

- **`app/domain`**: Pure logic and Pydantic models (Node, Edge, Chunk). Zero external dependencies.
- **`app/infrastructure`**: Concrete adapters for SurrealDB, Gemini API, and local embeddings. Optimized for CPU via **OpenVINO**.
- **`app/application`**: Use cases coordinating the ingestion and retrieval choreography.
- **`app/interfaces`**: FastAPI routers, request/response schemas, and the modern Web UI.

## Contributing

Before contributing, ensure all quality gates pass:

```bash
uv run ruff check app tests
uv run mypy app
uv run pytest
```

### Documentation

Documentation is managed via **Zensical**. To serve locally:

```bash
uv run zensical serve
```

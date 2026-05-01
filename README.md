# 📚 CodaCite

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hexagonal-orange.svg)](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
[![Database](https://img.shields.io/badge/Database-SurrealDB%20v3-red.svg)](https://surrealdb.com/)
[![Tooling](https://img.shields.io/badge/Tooling-uv%20%7C%20Podman-purple.svg)](https://github.com/astral-sh/uv)

**CodaCite** (Contextual Omni-Document Assistant with Cite-ability) is a state-of-the-art GraphRAG engine that transforms massive, unstructured document troves into a navigable, verifiable knowledge graph.

---

## 🚀 Quick Start

### 1. The Containerized Stack (Recommended)
Get the full environment (App + SurrealDB) running immediately:

```bash
# Start the SurrealDB v3 and CodaCite containers
podman-compose up -d --build

# Access the UI at http://localhost:8080
```

### 2. Manual Infrastructure
If you prefer to run the database separately:

```bash
# Start SurrealDB v3 with persistent storage
podman run --rm -p 8000:8000 \
  -v ./surreal_data:/var/lib/surrealdb \
  docker.io/surrealdb/surrealdb:v3.0.5 \
  start --user root --pass root surrealkv:///var/lib/surrealdb
```

### 3. Database Cleanup
If you need to wipe the database and start fresh:

#### Option A: Hard Reset (Recommended)
This removes all persistent data from the host.
```bash
# Stop the containers
podman-compose down

# Remove the data directory
rm -rf ./surreal_data

# Start fresh
podman-compose up -d
```

#### Option B: Soft Reset (SurrealQL)
Use this if you want to keep the container running but delete all data.
```bash
# Connect to the SurrealDB shell
podman exec -it surrealdb /surreal sql --endpoint http://localhost:8000 --user root --pass root

# Inside the shell, run:
REMOVE NAMESPACE codacite;
```

---

## 🛠️ Technology Stack

CodaCite utilizes a high-performance, local-first AI stack:

- **Core Orchestration**: Custom Hexagonal implementation for strict logic isolation.
- **Document Processing**: **Docling** (Layout-aware extraction), `langchain-text-splitters` (Recursive chunking).
- **Semantic Intelligence**: `fastcoref` (Linguistic resolution), **Gemini 2.0 Flash** (KG Extraction).
- **Agentic Orchestration**: **LangGraph** (Self-correcting RAG loop).
- **Vision AI**: `llama-cpp-python` (Local VLM for technical drawing descriptions).
- **Vector & Graph Store**: **SurrealDB v3** (Hybrid BM25 + HNSW Indexing + Graph Relations).
- **Embeddings**: `sentence-transformers` (BGE-M3 model).
- **Runtime**: `uv` (Package Management), `Podman` (Containerization).

---

## ⚙️ Local Development Setup

To run CodaCite directly on your host, ensure you maintain an isolated environment:

### 1. Environment Configuration
Add these to your `.bashrc` or session to protect your host from project-specific artifacts:

```bash
export UV_CACHE_DIR=$(pwd)/.uv_cache
export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv_python
```

### 2. Dependency Sync
```bash
# Install dependencies into project-local .venv
uv sync

# Download the BGE-M3 and LocalVLM model artifacts
uv run download-models
```

### 3. Run the App
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## 🏗️ The 8-Phase Ingestion Engine

CodaCite orchestrates an industrial-grade asynchronous pipeline:

1.  **Phase 1: Layout-Aware Loading**: **Docling** extracts text and identifies structural hierarchies (tables, headers).
2.  **Phase 2: Coreference Resolution**: **FastCoref** resolves linguistic ambiguities.
3.  **Phase 3: Recursive Chunking**: Dynamic splitting using semantic boundaries.
4.  **Phase 4: Multi-Model Persistence**: Chunks and metadata are committed to **SurrealDB**.
5.  **Phase 5: High-D Vectorization**: Generating 1024D embeddings via **BGE-M3**.
6.  **Phase 6: KG Extraction**: **Gemini** extracts structured Entities and Relationships.
7.  **Phase 7: Semantic Resolution**: Merging duplicate entities via vector and string similarity.
8.  **Phase 8: Graph Finalization**: Establishing notebook relationships and HNSW indexing.

---

## 🏗️ Architecture

- **`app/domain`**: Pure logic and Pydantic models. Zero external dependencies.
- **`app/infrastructure`**: Concrete adapters for SurrealDB, Gemini, and LocalVLM.
- **`app/application`**: Use cases coordinating the ingestion and retrieval choreography.
- **`app/interfaces`**: FastAPI routers and the modern Web UI.

---

## 🤖 Agentic Development

This project is managed by the **Antigravity** persona. All development is orchestrated via specialized workflows.

👉 **View the [Agent Guide](.agents/README.md)** for a full catalog of workflows and governance rules.

---

## 🧪 Quality Gates

```bash
uv run ruff check app tests  # Linting
uv run mypy app              # Type Safety
uv run pytest                # Functional Integrity
```

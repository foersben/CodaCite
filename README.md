# Enterprise Omni-Copilot

> GraphRAG-based Document Intelligence and Workflow Automation — built entirely in Python, served as a containerised FastAPI application, published to the GitHub Container Registry on every merge.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [System Architecture](#system-architecture)
   - [Module Overview](#module-overview)
   - [Data Flow](#data-flow)
3. [Technology Choices and Rationale](#technology-choices-and-rationale)
4. [Web Interface](#web-interface)
5. [REST API](#rest-api)
6. [CI/CD Pipeline](#cicd-pipeline)
   - [Test & Lint Job](#test--lint-job)
   - [Docker Build & Publish Job](#docker-build--publish-job)
   - [Tagging Strategy](#tagging-strategy)
7. [Container Strategy](#container-strategy)
   - [Why Models Are Not Baked Into the Image](#why-models-are-not-baked-into-the-image)
   - [Running With Docker Compose](#running-with-docker-compose)
   - [Pre-downloading the Embedding Model](#pre-downloading-the-embedding-model)
8. [Local Development](#local-development)
9. [Configuration Reference](#configuration-reference)
10. [Quality Gates](#quality-gates)

---

## What This Is

Enterprise Omni-Copilot is a self-hosted, privacy-preserving document intelligence system. Upload your PDFs, Word documents, or Markdown files and ask questions — the system extracts entities, builds a knowledge graph, embeds every text chunk locally, and retrieves answers through a hybrid vector-plus-graph search pipeline.

All ML inference (embeddings and cross-encoder reranking) runs on-device. No document text is sent to a third party unless you explicitly configure an OpenAI-compatible LLM for intent routing.

---

## System Architecture

```
                         ┌──────────────────────────────────┐
                         │         FastAPI Application        │
                         │  GET /           (web UI)          │
                         │  POST /ui/ingest (web UI)          │
                         │  POST /ui/query  (web UI)          │
                         │  POST /api/v1/ingest               │
                         │  POST /api/v1/query                │
                         │  GET  /health                      │
                         └─────────────┬────────────────────-─┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐            ┌───────▼──────┐          ┌────────▼────────┐
    │  Ingestion   │            │  Embeddings  │          │     Agents      │
    │  loader      │            │  LocalEmbedder│          │  IntentRouter   │
    │  preprocessor│            │  (BAAI/bge)  │          │  (LLM prompt)   │
    │  chunker     │            └───────┬──────┘          └────────┬────────┘
    └──────┬──────┘                    │                           │
           │                           │                           │
    ┌──────▼──────────────────────────-┼───────────────────────────▼────────┐
    │                            Graph / Store                               │
    │                           SurrealDB                                    │
    │              (vector index + entity graph + relationships)             │
    └───────────────────────────────────┬───────────────────────────────────┘
                                        │
                               ┌────────▼────────┐
                               │    Retrieval     │
                               │  HybridRetriever │
                               │  (vector search  │
                               │  → graph travel  │
                               │  → reranking)    │
                               └─────────────────-┘
```

### Module Overview

| Module | Package | Responsibility |
|--------|---------|---------------|
| **Ingestion** | `app/ingestion/` | Load raw documents (PDF via `pypdf`, DOCX via `python-docx`, Markdown natively), normalise Unicode, strip noise, split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` |
| **Embeddings** | `app/embeddings/` | Wrap a locally-loaded `SentenceTransformer` (`BAAI/bge-large-en-v1.5`) to produce 1024-dimensional dense vectors; never makes network calls at runtime |
| **Graph** | `app/graph/` | Extract named entities and relationships from chunk text via an LLM; persist chunks (with their embeddings), entities, and relationships to SurrealDB |
| **Retrieval** | `app/retrieval/` | Three-stage pipeline — (1) cosine vector search in SurrealDB, (2) 1–2 hop graph traversal to enrich the candidate pool, (3) cross-encoder reranking to score and sort the final top-K results |
| **Agents** | `app/agents/` | LLM-based intent router that classifies every query as `knowledge_retrieval` or `action_execution` before the retrieval pipeline runs, enabling future action dispatch |

### Data Flow

**Ingest path**

```
Upload file
  → DocumentLoader   (parse to plain text)
  → TextPreprocessor (normalise Unicode, collapse whitespace)
  → TextChunker      (1 024-char chunks, 128-char overlap)
  → LocalEmbedder    (dense vector per chunk)
  → GraphStore       (store chunk + embedding in SurrealDB)
  → EntityExtractor  (LLM → entities + relationships)
  → GraphStore       (store entity nodes + relationship edges)
```

**Query path**

```
User query
  → IntentRouter     (LLM classifies intent)
  → LocalEmbedder    (embed query)
  → GraphStore       (vector search, top-2K candidates)
  → GraphStore       (graph traversal, depth 2)
  → CrossEncoderReranker (score all candidates)
  → top-K results    (returned to caller)
```

---

## Technology Choices and Rationale

### FastAPI

FastAPI is the application framework for both the JSON API and the web interface. The choice was deliberate:

- **Async-first**: ingestion and retrieval involve awaitable I/O (database, file reads) — FastAPI's native async support handles this without thread-pool overhead.
- **Pydantic models everywhere**: request bodies, response bodies, and application settings are all validated by Pydantic v2. This eliminates an entire class of silent type-mismatch bugs.
- **Zero-cost OpenAPI**: interactive `/docs` and `/redoc` endpoints are generated automatically from the same type annotations. No separate documentation step.
- **Factory pattern (`create_app`)**: the app is created by a function rather than at module import time. Each test gets a fresh, isolated instance without shared mutable state, which was important for keeping the 57-test suite deterministic.

### SurrealDB

SurrealDB is used as the single backend store for both vectors and the entity graph. A traditional architecture would require a dedicated vector database (e.g. Qdrant, Weaviate) plus a graph database (Neo4j) — two services to deploy, two connection pools to manage, two backup strategies. SurrealDB provides native vector indexing, a graph traversal query language, and schemaless records in one process. The `docker-compose.yml` runs it in `memory` mode for development; a persistent mode would use a volume-mounted `rocksdb` backend.

### BAAI/bge-large-en-v1.5

This model was chosen because:

1. **1024-dim vectors** give strong recall on technical and enterprise English text.
2. **It is free and weights are public** — no API key required, no per-token cost, no data leaving the host.
3. **The BGE family is optimised for asymmetric search** (short query vs. longer passage), which matches our use case exactly.

The model is downloaded once to `./models/BAAI/bge-large-en-v1.5` and volume-mounted into the container at runtime so it persists across image updates (see [Why Models Are Not Baked Into the Image](#why-models-are-not-baked-into-the-image)).

### LangChain (chunking only)

LangChain is used exclusively for `RecursiveCharacterTextSplitter`. This splitter attempts to break text on `\n\n`, `\n`, ` `, and finally individual characters in order of preference, which preserves paragraph and sentence boundaries far better than a naive sliding window. No LangChain agents, chains, or memory abstractions are used — this keeps the dependency surface minimal and avoids the frequent breaking changes in LangChain's higher-level APIs.

### uv (dependency and runtime manager)

`uv` replaces `pip` + `virtualenv` + `pip-tools`. It resolves and installs the full dependency tree in seconds rather than minutes, produces a deterministic lock file, and is available as a single binary that is copied into the Docker image from the official `ghcr.io/astral-sh/uv:latest` image — no `pip install uv` step needed in CI or Docker.

---

## Web Interface

The user interface is served directly by FastAPI using **Jinja2 server-side rendering**. There is **zero client-side JavaScript**.

This was a deliberate design constraint. Every interaction that a user needs — uploading a file, submitting a query, seeing results — is achievable with plain HTML `<form>` elements and HTTP POST. Server-side rendering means:

- The UI works in any browser, including text-only browsers and browsers with JavaScript disabled.
- There is no build step, no node_modules, no bundler.
- Template security (XSS prevention) is handled by Jinja2's auto-escaping, not by a client-side sanitisation library.
- The state of the page after a query is fully representable in a URL (via `?message=` and `?error=` query params on the redirect after ingest), which means the browser back button works correctly.

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Render the main page. Accepts `?message=` and `?error=1` query params (set by ingest redirect) to display flash messages. |
| `POST` | `/ui/ingest` | Receive a multipart file upload; run the full ingest pipeline; redirect back to `/` with a summary message (PRG pattern). |
| `POST` | `/ui/query` | Receive a form-encoded query + `top_k`; run the full retrieval pipeline; render the results inline on the same page. |

The Post-Redirect-Get (PRG) pattern is used for the ingest form specifically to prevent the browser from re-submitting the file upload on a page refresh.

UI routes are excluded from the OpenAPI schema (`include_in_schema=False`) to keep the JSON API docs clean.

---

## REST API

The JSON API is versioned under `/api/v1/`. All request and response bodies are validated by Pydantic and documented at `/docs`.

### `POST /api/v1/ingest`

Upload a document for ingestion. Accepted formats: `.pdf`, `.docx`, `.md`, `.markdown`.

**Request**: `multipart/form-data`, field `file`.

**Response `200`**:
```json
{
  "filename": "report.pdf",
  "chunks_processed": 42,
  "entities_extracted": 17
}
```

### `POST /api/v1/query`

Run a hybrid GraphRAG query against the knowledge base.

**Request body**:
```json
{
  "query": "What are the key findings in Q3?",
  "top_k": 5
}
```

**Response `200`**:
```json
{
  "query": "What are the key findings in Q3?",
  "intent": "knowledge_retrieval",
  "results": [
    {
      "text": "Q3 revenue increased by 12% year-on-year...",
      "score": 0.934,
      "source": "report.pdf"
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}`. Used by Docker healthchecks and uptime monitors.

---

## CI/CD Pipeline

The pipeline lives in `.github/workflows/ci.yml` and has two jobs with a strict dependency: **`docker` only runs if `test` passes**.

```
push / pull_request
       │
       ▼
  ┌─────────┐
  │  test   │  ← runs on every push and PR
  └────┬────┘
       │ needs: test
       ▼
  ┌─────────┐
  │  docker │  ← builds always; pushes only on non-PR events
  └─────────┘
```

### Test & Lint Job

Runs on every push (all branches) and every pull request targeting `main` / `master`.

1. **Install `uv`** — fetched as a pre-built binary using the official `astral-sh/setup-uv` action; no `pip` invocation needed.
2. **`uv sync --extra dev`** — installs the full dependency tree including test and lint tools.
3. **Cache the embedding model** — the `BAAI/bge-large-en-v1.5` weights are ~1.3 GB. The `actions/cache` step keys on `models-baai-bge-large-en-v1.5-{OS}` so the download only happens once per runner OS, then the weights are served from cache on subsequent runs.
4. **`uv run pytest --cov=app tests/`** — runs the test suite with coverage; the XML report is uploaded as a workflow artifact on every run (including failures) for inspection.
5. **`uv run ruff check app/ tests/`** — fast linting covering PEP 8 style (E/W), unused imports (F), import ordering (I), naming (N), modern syntax upgrades (UP), bug-prone patterns (B), comprehension optimisations (C4), and simplification opportunities (SIM).
6. **`uv run mypy app/`** — strict type checking with `ignore_missing_imports = true` so that third-party libraries without stubs do not block the check.

### Docker Build & Publish Job

Runs after `test` passes. Triggered on pushes to any branch, semver tags (`v*.*.*`), and pull requests (build-only, no push).

1. **Docker metadata** (`docker/metadata-action`) — computes the set of image tags and Open Container Initiative (OCI) labels from the Git context. No hardcoded version strings anywhere in the workflow.
2. **QEMU + Buildx** — enables cross-compilation of `linux/arm64` on an `x86_64` GitHub-hosted runner. A single `docker build-push-action` step produces a multi-platform manifest.
3. **Login to GHCR** — uses `${{ secrets.GITHUB_TOKEN }}` with `packages: write` permission. No external credentials are required; the token is automatically provided by GitHub Actions.
4. **Build and push** — layer cache is stored in the GitHub Actions cache (`cache-from: type=gha`, `cache-to: type=gha,mode=max`). On pull requests, `push: false` ensures no image is published from an unreviewed branch.

### Tagging Strategy

| Trigger | Example tags produced |
|---------|----------------------|
| Push to `main` | `latest`, `main`, `sha-abc1234` |
| Push to feature branch | `my-feature-branch`, `sha-abc1234` |
| Push of tag `v1.2.3` | `1.2.3`, `1.2`, `sha-abc1234` |
| Pull request | `pr-42`, `sha-abc1234` (not pushed) |

The `latest` tag is only applied when `is_default_branch` is true, so `latest` always refers to the most recently merged code on `main`/`master`.

---

## Container Strategy

### Why Models Are Not Baked Into the Image

The `BAAI/bge-large-en-v1.5` model weighs approximately 1.3 GB. Earlier versions of the Dockerfile downloaded the model weights during `docker build`. This had three problems:

1. **Build time**: every CI run that missed the layer cache triggered a 1.3 GB download, adding several minutes to the pipeline.
2. **Image size**: the published image was unnecessarily large. Anyone pulling it for the first time would download 1.3 GB of model weights they might already have locally.
3. **Update coupling**: updating the model required rebuilding and republishing the image even if no application code changed.

The current approach volume-mounts the model directory at runtime:

```yaml
volumes:
  - ./models:/app/models:ro
```

The models directory is populated once by running:

```bash
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  ghcr.io/foersben/ml_reply_example:latest \
  uv run python scripts/download_models.py
```

After that, the models persist on the host and survive any number of image updates.

### Running With Docker Compose

```bash
# 1. Pre-download model weights (first time only)
mkdir -p models
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  ghcr.io/foersben/ml_reply_example:latest \
  uv run python scripts/download_models.py

# 2. Start the full stack (SurrealDB + app)
docker compose up
```

The app will be available at:

- **Web UI** → http://localhost:8080
- **Interactive API docs** → http://localhost:8080/docs
- **ReDoc** → http://localhost:8080/redoc
- **Health check** → http://localhost:8080/health
- **SurrealDB** → ws://localhost:8000/rpc

`docker-compose.yml` configures a healthcheck on SurrealDB (`/surreal isready`) and sets `depends_on: condition: service_healthy` on the app, so the application only starts once the database is accepting connections.

### Pulling the Published Image

```bash
docker pull ghcr.io/foersben/ml_reply_example:latest
```

Multi-platform manifest covers `linux/amd64` and `linux/arm64` (Apple Silicon, AWS Graviton, Raspberry Pi 4).

---

## Local Development

**Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install all dependencies (including dev tools)
uv sync --extra dev

# Download the embedding model (first time only)
uv run python scripts/download_models.py

# Start a local SurrealDB (requires Docker)
docker run -d --name surrealdb -p 8000:8000 \
  surrealdb/surrealdb:latest \
  start --log trace --user root --pass root memory

# Run the application
uv run uvicorn app.main:app --reload --port 8080
```

### Run Tests

```bash
uv run pytest --cov=app tests/ --cov-report=term-missing
```

### Lint

```bash
uv run ruff check app/ tests/
```

### Type-check

```bash
uv run mypy app/
```

---

## Configuration Reference

All settings are read from environment variables (or an optional `.env` file in the project root). Pydantic Settings validates types and ranges at startup, so misconfiguration fails fast with a clear error message.

| Variable | Default | Description |
|----------|---------|-------------|
| `SURREALDB_URL` | `ws://localhost:8000/rpc` | WebSocket URL of the SurrealDB instance |
| `SURREALDB_USER` | `root` | SurrealDB username |
| `SURREALDB_PASS` | `root` | SurrealDB password |
| `SURREALDB_NS` | `omni` | SurrealDB namespace |
| `SURREALDB_DB` | `copilot` | SurrealDB database name |
| `MODELS_DIR` | `./models` | Path to the directory containing downloaded model weights |
| `EMBEDDING_MODEL_ID` | `BAAI/bge-large-en-v1.5` | Sub-path inside `MODELS_DIR` for the embedding model |
| `CHUNK_SIZE` | `1024` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `128` | Overlap in characters between adjacent chunks |
| `OPENAI_API_KEY` | *(empty)* | API key for an OpenAI-compatible LLM (used by intent router and entity extractor; leave blank to use mock behaviour in tests) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model identifier passed to the LLM API |

---

## Quality Gates

Every pull request must pass all of the following before the Docker image is published:

| Check | Tool | What it catches |
|-------|------|----------------|
| Unit + integration tests | `pytest` + `pytest-asyncio` | Regressions in all five modules; async I/O correctness |
| Code coverage report | `pytest-cov` | Uploaded as a workflow artifact for review |
| Style and correctness lint | `ruff` | Unused imports, shadowed names, unreachable code, non-idiomatic patterns |
| Static type checking | `mypy --strict` | Type mismatches, missing return types, incorrect argument types |
| Container build validation | `docker/build-push-action` (PR mode) | Dockerfile syntax errors, missing files, broken `RUN` commands |

# CodaCite: GraphRAG-based Document Intelligence

CodaCite is a high-performance, local-first RAG (Retrieval-Augmented Generation) engine designed to transform static documents into dynamic, graph-linked knowledge bases. Built for speed, precision, and verifiability, it leverages **SurrealDB** as a hybrid graph-vector store to enable deep-context retrieval with character-level citation accuracy.

## 🚀 Vision

To provide a private, scalable alternative to cloud-based document AI, focusing on:
1. **Verifiability**: Every response must be grounded in explicit character offsets.
2. **Contextual Intelligence**: Using Graph relationships to navigate complex document hierarchies.
3. **Local Sovereignty**: Zero-data-leakage architecture running on consumer-grade hardware.

## 🏗️ Architecture: Vertical Slices

CodaCite is built using a **Vertical Slice Architecture** (Modular Monolith). Instead of horizontal layers, the system is organized into autonomous, feature-oriented pipelines.

*   **Ingestion**: Structural chunking, metadata extraction, and graph linking.
*   **Retrieval**: Hybrid vector-graph search with notebook-level scoping.
*   **Generation**: Citations-first reasoning using state-of-the-art LLMs.
*   **Core**: Shared infrastructure, Dependency Injection, and Global Config.

## 🌟 Key Features

- **Deep Grounding**: Verifiable response generation with verbatim quotes and character-offset citations.
- **Structural Context**: Deterministic chunking that preserves document hierarchy and provenance.
- **Graph Scoping**: Instantaneous context switching through graph-enforced "Notebook" isolation.
- **Autonomous Resolution**: Built-in entity resolution and semantic merging pipelines.

## 🛠️ Tech Stack

*   **Runtime**: Python 3.13+ (managed by `uv`)
*   **Database**: SurrealDB v3.0+ (O(1) Direct ID Retrieval, Hybrid Graph-Vector)
*   **Intelligence**:
    *   **Embeddings**: High-density local embeddings (FastEmbed/Transformers).
    *   **Generation**: DeepSeek-R1 (for verifiable grounding) and Gemini (multimodal).
    *   **Orchestration**: `LangGraph` for stateful reasoning.
*   **Infrastructure**: Podman & Podman-Compose.

## 📦 Getting Started

### 1. Prerequisites
Install `uv` (modern Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/foersben/codacite
cd codacite

# Initialize virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 3. Spin up Infrastructure
We use `podman` for zero-root database isolation:
```bash
podman-compose up -d
```

### 4. Run the Engine
```bash
uv run python -m app.main
```

## 🤖 Agent Workspace

CodaCite is built with **Agentic Workflows** in mind. Developers can use the following slash commands in an Antigravity-supported IDE:

- `/bootstrap`: Rebuild environment from scratch.
- `/qa-pass <file>`: Generate and verify tests for a specific file.
- `/qa-pass-all`: Audit and generate coverage for the entire codebase.
- `/implement`: Plan and implement a new feature.
- `/commit`: Standardized pre-commit checks and commit generation.
- `/coverage-boost`: Target and fill specific testing gaps.
- `/purge-cruft`: Iteratively remove legacy code and redundant wrappers.
- `/update-readme`: Synchronize this file with current architecture.

## 📜 Documentation

For deep dives into the system, refer to the [Docs directory](./docs/):
- [Architecture](./docs/architecture.md)
- [Concepts](./docs/concepts.md)
- [Data Pipeline](./docs/data_pipeline.md)

---
© 2024 CodaCite Contributors. Released under the MIT License.

# 📚 CodaCite: The GraphRAG Textbook

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Vertical_Slice-green.svg)](docs/architecture.md)
[![Database](https://img.shields.io/badge/Database-SurrealDB%20v3-red.svg)](https://surrealdb.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CodaCite** (Contextual Omni-Document Assistant with Cite-ability) is a formal, pedagogical implementation of a **GraphRAG-based Document Intelligence System**. It is designed not just as a tool, but as a "living textbook" on the physics of modern AI retrieval and ingestion.

---

## 🏛️ Architectural Philosophy

CodaCite has transitioned from a traditional Hexagonal layer-cake to a **Vertical Slice Architecture**. This modular monolith design organizes code around **features** rather than technical layers, ensuring that the business logic for Ingestion, Retrieval, and Extraction remains autonomous and highly maintainable.

For a deep dive into the architectural mechanics, see [Chapter 1: Architectural Paradigms](docs/architecture.md).

---

## 🚀 Quick Start (The Laboratory)

To explore the system in a controlled, containerized environment:

```bash
# 1. Download necessary model artifacts
uv run download-models

# 2. Start the Orchestration (App + SurrealDB)
podman-compose up -d --build

# 3. Access the UI at http://localhost:8080
```

---

## 📖 The "Textbook" Curriculum

[The documentation](https://foersben.github.io/CodaCite/) is organized as a sequential curriculum for engineers and researchers:

1. **[Preface: The Syllabus](https://foersben.github.io/CodaCite/)** - Introduction to the system and the learning objectives.
2. **[Chapter 1: Vertical Slice Architecture](https://foersben.github.io/CodaCite/architecture/)** - Understanding the feature-oriented modular monolith.
3. **[Chapter 2: The Ingestion Lifecycle](https://foersben.github.io/CodaCite/data_pipeline/)** - 9-Phase transformation with `Docling` and memory-efficient `BGE-M3` pipelines.
4. **[Chapter 3: Search & Retrieval Mechanics](https://foersben.github.io/CodaCite/retrieval/)** - Analysis of Hybrid Search and **Adaptive Intent Routing** for instant global summaries.
5. **[Chapter 4: Infrastructure & Persistence](https://foersben.github.io/CodaCite/infrastructure/)** - How SurrealDB and local model quantization (GGUF/INT8) power the engine.
6. **[Chapter 5: Interface Design](https://foersben.github.io/CodaCite/ui/)** - Exploring functional density and the 1.5x scaling system.
7. **[Chapter 6: Operations & Quality](https://foersben.github.io/CodaCite/operations/)** - The CI/CD pipelines and deployment strategies.
8. **[Appendix A: Developer Context](https://foersben.github.io/CodaCite/AGENT_CONTEXT/)** - Implementation-level quirks and troubleshooting.

---

## 🛠️ The Local-First Intelligence Stack

CodaCite utilizes a high-performance, private-first stack:

* **Vector Engine**: `BGE-M3` (Semantic Chunking & Embedding).
* **Local Reasoning**: `DeepSeek-R1` (GGUF via llama.cpp) and `Gemini 2.0 Flash`.
* **Reranker**: `ModernBERT` (INT8 Quantized via OpenVINO).
* **Extraction**: `GLiNER` (Zero-shot NER) and `FastCoref`.
* **Persistence**: **SurrealDB v3.0.5** (Graph-Vector Hybrid).
* **Orchestration**: **LangGraph** (Agentic Retrieval Loops).

---

## 🤖 Agent Workspace

CodaCite is built for **Agentic Development**. The following slash commands are available to Antigravity and other compatible agents to automate high-frequency tasks:

| Command | Description |
| :--- | :--- |
| `/implement` | Safely implements a new feature from planning to testing. |
| `/document-all` | Batch updates inline docstrings and global project documentation. |
| `/sync-zensical` | Synchronizes the documentation tracking configuration. |
| `/qa-pass` | Generates and verifies unit tests for a specific target file. |
| `/commit` | Runs pre-commit checks and handles branch synchronization. |
| `/refactor-all` | Safely executes a codebase-wide refactor folder-by-folder. |
| `/run-tests` | Runs the standard linting and functional test suite. |

---

## 🧪 Quality Gates

```bash
uv run ruff check app tests  # Linguistic & Structural Linting
uv run mypy app              # Strict Mathematical Type Safety
uv run pytest                # Functional Integrity Verification
```

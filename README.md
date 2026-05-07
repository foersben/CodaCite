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
# 1. Start the Orchestration (App + SurrealDB)
podman-compose up -d --build

# 2. Access the UI at http://localhost:8080
# Note: The first launch triggers the 'Cognitive Bootstrap,' downloading ~5GB of AI models.
```

---

## 📖 The "Textbook" Curriculum

The documentation is organized as a sequential curriculum for engineers and researchers:

1. **[Preface: The Syllabus](docs/index.md)** - Introduction to the system and the learning objectives.
2. **[Chapter 1: Vertical Slice Architecture](docs/architecture.md)** - Understanding the feature-oriented modular monolith.
3. **[Chapter 2: The Ingestion Lifecycle](docs/data_pipeline.md)** - A deep dive into the 8-Phase transformation from text to graph.
4. **[Chapter 3: Search & Retrieval Mechanics](docs/retrieval.md)** - Analysis of Hybrid Search and agentic self-correction.
5. **[Chapter 4: Infrastructure & Persistence](docs/infrastructure.md)** - How SurrealDB and local model quantization power the engine.
6. **[Chapter 5: Interface Design](docs/ui.md)** - Exploring functional density and the 1.5x scaling system.
7. **[Chapter 6: Operations & Quality](docs/operations.md)** - The CI/CD pipelines and deployment strategies.
8. **[Appendix A: Developer Context](docs/AGENT_CONTEXT.md)** - Implementation-level quirks and troubleshooting.

---

## 🛠️ The Local-First Intelligence Stack

CodaCite utilizes a high-performance, private-first stack:

* **Vector Engine**: `BGE-M3` (Semantic Chunking & Embedding).
* **Reasoning Agent**: `Gemini 2.0 Flash` (with local `GLiNER` fallback).
* **Reranker**: `ModernBERT` (INT8 Quantized via OpenVINO).
* **Persistence**: **SurrealDB v3.0.5** (Graph-Vector Hybrid).
* **Orchestration**: **LangGraph** (Agentic Retrieval Loops).

---

## 🧪 Quality Gates

```bash
uv run ruff check app tests  # Linguistic & Structural Linting
uv run mypy app              # Strict Mathematical Type Safety
uv run pytest                # Functional Integrity Verification
```

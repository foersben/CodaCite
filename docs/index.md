# Preface: Methodology & Syllabus

Welcome to the **CodaCite Textbook**, the authoritative technical guide for the GraphRAG-based Document Intelligence platform. This documentation is designed to be read sequentially, providing a pedagogical transition from high-level architectural intent to granular implementation details.

## The CodaCite Manifesto

In an era of increasingly opaque AI systems, CodaCite is built upon three non-negotiable pillars:

1. **Absolute Provenance**: Every AI-generated claim is anchored to a specific character offset (`start_char`, `end_char`) in the source PDF.

2. **Local Sovereignty**: All inference (LLM, Embeddings, OCR) is executed on-premises via Podman, ensuring zero data leakage.

3. **Graph-Augmented Retrieval**: Relationships are not just stored; they are traversed to provide context that traditional vector search misses.

4. **Self-Correction**: The system does not merely "search"; it reasons about its own retrieval quality, rewriting queries and grading context until it reaches the precision required for high-stakes analysis.

## The Syllabus

This "textbook" is organized into the following chapters:

* **Chapter 1: [System Architecture](architecture.md)** — Explores the "Vertical Slice" methodology and our modular monolith design.

* **Chapter 2: [The Data Ingestion Lifecycle](data_pipeline.md)** — A deep dive into the 8-phase transformation from raw text to structured knowledge.

* **Chapter 3: [Search and Retrieval Mechanics](retrieval.md)** — Details the physics of hybrid search and the LangGraph self-correction loop.

* **Chapter 4: [Infrastructure and Foundation](infrastructure.md)** — Examines the role of SurrealDB and local model quantization.

* **Chapter 5: [The User Interface](ui.md)** — Discusses the UX philosophy of Notebook-scoped analysis.

* **Chapter 6: [Operations & Quality Gates](operations.md)** — Details the CI/CD pipeline and container orchestration.

* **Appendix A: [Developer Context](agent_context.md)** — Implementation heuristics and troubleshooting for AI agents.

---

> [!NOTE]
> This documentation is a living artifact. All architectural changes must be reflected here to maintain the system's "textbook" integrity.

# Architecture

The architecture of CodaCite represents a sophisticated synthesis of modern data processing paradigms, unified under a rigorous Hexagonal Architecture. This structural philosophy, also known as the Ports and Adapters pattern, strictly isolates the core business logic from the volatile external dependencies of databases, user interfaces, and third-party models. By defining explicit contractual interfaces, the system guarantees that the intricate logic governing knowledge extraction and retrieval remains pristine and universally testable. This decoupling empowers the engineering team to swap out underlying technologies, such as transitioning between local embedding models and cloud-based inferential engines, without inducing cascading failures throughout the application codebase.

The entry point for all interactions within this ecosystem is managed by a high-performance, asynchronous gateway powered by FastAPI. This application programming interface layer is responsible for receiving varied forms of unstructured documents and complex user queries, validating the inbound payloads against strict schemas before delegating the tasks deeper into the system. It acts as a resilient shield, absorbing concurrent traffic spikes and ensuring that only well-formed data enters the processing pipeline. By abstracting the network protocols and serialization concerns, the gateway allows the subsequent layers to focus entirely on the profound work of semantic orchestration.

A critical refinement in the CodaCite architecture is the introduction of **Multi-Notebook Orchestration**. This layer allows users to partition their knowledge base into discrete, manageable containers called "Notebooks." Rather than operating on a monolithic document store, the system utilizes graph-based relations to dynamically filter context during search and retrieval. When a document is ingested, it is linked to one or more notebooks via `belongs_to` graph edges. This enables high-performance, responsive UI interactions where users can select or deselect specific notebooks to instantly scope the AI's "active memory" during a chat session. **Scoping is enforced at the database level**, ensuring that vector searches only consider chunks associated with the active notebook set.


The foundational bedrock of this architecture is provided by **SurrealDB**, a multi-model database engine equipped with **Hybrid Indexing** capabilities. This infrastructural layer transcends the limitations of traditional relational stores by naturally representing the complex, multi-dimensional reality of the ingested data. It combines Hierarchical Navigable Small World (**HNSW**) vector indexing for semantic similarity with **BM25** full-text search for exact keyword matching. This convergence of vector mathematics, full-text retrieval, and graph theory within a single persistent store is the critical enabler of the system's ability to reason across vast troves of unstructured enterprise knowledge.

## The 8-Phase Ingestion Pipeline

CodaCite orchestrates a rigorous, asynchronous pipeline to decompose documents into a high-fidelity knowledge graph:

1. **Phase 1: Loading & Preprocessing**: File validation, normalization, and text extraction (PDF/Text).
2. **Phase 2: Coreference Resolution**: Uses `fastcoref` to normalize linguistic references (e.g., resolving "he" to "Albert Einstein").
3. **Phase 3: Recursive Chunking**: Partitions the resolved text into overlapping semantic fragments using `RecursiveCharacterTextSplitter`.
4. **Phase 4: Document Persistence**: Commits raw text chunks and establishes `document -> belongs_to -> notebook` relations in SurrealDB.
5. **Phase 5: Vectorization (Embedding)**: Generates 1024D vectors for every chunk using the BGE-M3 model (optimized via OpenVINO).
6. **Phase 6: Knowledge Extraction**: Discovery of entity Nodes and relationship Edges from chunks using Google Gemini (or GLiNER fallback).
7. **Phase 7: Entity Resolution**: Deduplicates extracted nodes against the global graph using Jaro-Winkler similarity and vector distance.
8. **Phase 8: Finalization**: Updates the document status to `active` and triggers maintenance on the vector index.

## Ingestion Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI (Router)
    participant UC as IngestionUseCase
    participant DB as SurrealDB (Store)
    participant AI as AI Models (BGE/Gemini)

    U->>API: Upload Document
    API->>API: Phase 1: Load & Preprocess (Sync)
    API->>UC: ingest_and_queue(text, filename)
    UC->>DB: Phase 1: Save Initial Document Record
    UC-->>API: document_id
    API-->>U: 202 Accepted (Processing)

    Note over UC,AI: Background Processing (Asynchronous)
    API->>UC: add_task: process_background(doc_id, text)
    UC->>AI: Phase 2: Resolve Coreferences
    AI-->>UC: Resolved Text
    UC->>UC: Phase 3: Recursive Chunking
    UC->>DB: Phase 4: Save Raw Chunks
    UC->>AI: Phase 5: Embed Chunks (Batch)
    AI-->>UC: Vectors
    UC->>AI: Phase 6: Extract Graph Fragments
    AI-->>UC: Nodes & Edges
    UC->>UC: Phase 7: Resolve/Deduplicate Entities
    UC->>DB: Phase 8: Finalize Status (Active)
```

## The Agentic RAG Retrieval Pipeline

The retrieval logic is orchestrated via a self-correcting **LangGraph** loop, ensuring high precision and recall through iterative refinement:

1. **Stage 1: Hybrid Retrieval**: Executes a parallel BM25 and HNSW search, combined with multi-hop graph traversal.
2. **Stage 2: Relevance Grading**: Uses a local LLM to grade each retrieved snippet.
3. **Stage 3: Iterative Rewriting**: Rephrases the query if retrieval results are insufficient.
4. **Stage 4: Context Synthesis**: Reranks and aggregates verified evidence for the final generation.

```mermaid
graph TD
    UI[Web UI / Notebooks]
    API[FastAPI Gateway]
    APP[LangGraph Agentic Loop]
    DOMAIN[Domain Models & Ports]
    INFRA[Infrastructure Adapters]
    DB[(SurrealDB: Hybrid + Graph)]
    MODELS[NLP Models]

    UI --> API
    API --> APP
    APP --> DOMAIN
    APP --> INFRA
    INFRA --> DB
    INFRA --> MODELS

    subgraph "Graph Scoping"
        DOC[Document] -- belongs_to --> NB[Notebook]
        CH[Chunk] -- extracted_from --> DOC
    end
```

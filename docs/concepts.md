# Core Concepts & Architectural Pillars

This document provides a rigorous deep-dive into the foundational principles that govern the CodaCite engine. It explains the rationale behind our technical choices, the underlying mechanisms, and the trade-offs inherent in building a high-precision, agentic RAG system.

---

## 1. Vertical Slice Architecture

### Rationale: Vertical Slice

Traditional N-tier or Hexagonal architectures often lead to "Layer Debt," where a simple feature change requires touching five different directories (Domain, Infrastructure, Application, Interfaces, etc.). In an AI-driven project like CodaCite, where the boundary between "Data" and "Logic" is fluid, we chose **Vertical Slice Architecture** to prioritize feature cohesion over technical categorization.

### Mechanism: Feature Slices

Instead of horizontal layers, the application is divided into vertical slices located in `app/pipelines/<slice_name>/`. Each slice contains:

* **Domain Models**: Pydantic/TypedDict definitions specific to that feature.

* **Business Logic**: The core functions or LangGraph nodes.

* **Infrastructure Adapters**: Specific database queries or API clients needed for that slice.

### Trade-offs: Modularity vs. Duplication

* **Advantage**: Extreme modularity. You can refactor or delete the "Ingestion" slice without risking the "Generation" slice's stability.

* **Advantage**: Reduced cognitive load. Developers only need to look at one folder to understand a feature's lifecycle.

* **Disadvantage**: Potential for minor duplication in utility code (mitigated by a shared `app/core` layer).

* **Disadvantage**: Requires discipline to prevent slices from becoming tightly coupled.

### Value Contribution: Maintainability

By reducing "spaghetti dependencies," we ensure that the codebase remains maintainable even as AI models and APIs evolve rapidly. It allows us to ship specialized optimizations (like VRAM routing) within a slice without affecting the rest of the system.

---

## 2. Agentic RAG with LangGraph

### Rationale: Dynamic Control

Linear RAG (Retrieve -> Augment -> Generate) is brittle. It fails when a query is ambiguous or when the retrieved documents are irrelevant. We implemented an **Agentic Loop** using LangGraph to allow the system to "think" before and after generating an answer.

### Mechanism: State Machine

The system treats the RAG process as a state machine (`RAGState`). A graph of nodes (functions) processes the state:

1. **Query Rewrite**: Analyzes the user's intent and expands the query for better retrieval.

2. **Retrieve & Rerank**: Fetches candidates and uses a cross-encoder to score them.

3. **Self-Correction**: If the reranker scores are too low, the agent can cycle back to rewrite the query.

4. **Evidence Generation**: Produces an answer only when high-confidence grounding is found.

### Trade-offs: Precision vs. Latency

* **Advantage**: Higher accuracy and "hallucination resistance" through self-reflection.

* **Advantage**: Observability. Every step of the "thought process" is a node in the graph that can be logged and audited.

* **Disadvantage**: Increased latency. Multiple LLM calls for query analysis and reflection take more time.

* **Disadvantage**: Complexity. Managing the state transitions and generic bounds of the graph requires high technical overhead.

### Value Contribution: Grounded Intelligence

This provides "Evidence-Based Intelligence." Users can trust CodaCite because it doesn't just guess; it iterates until it finds verifiable proof in the provided sources.

---

## 3. SurrealDB Multi-Model Persistence

### Rationale: Unified Engine

Most RAG systems require three separate databases: a Relational DB for metadata, a Vector DB for embeddings, and a Graph DB for citations. We chose **SurrealDB** because it natively supports all three models in a single, unified engine.

### Mechanism: Record-Level Graphing

SurrealDB stores data as "Nodes" (e.g., `Document`, `Chunk`, `Entity`) and "Edges" (e.g., `extracted_from`, `belongs_to`).

* **Vector Search**: Uses HNSW indices directly on record fields for O(1) similarity matching.

* **Graph Relations**: Allows complex traversals (e.g., "Find all chunks related to 'Project X' that were mentioned in this specific notebook").

### Trade-offs: Integration vs. Maturity

* **Advantage**: Simplified Infrastructure. One binary to manage, back up, and secure.

* **Advantage**: Record-Level Graph Access. We can fetch related data without expensive JOIN operations.

* **Disadvantage**: Maturing Ecosystem. The Python SDK and documentation are evolving rapidly, requiring strict adherence to version-specific patterns.

* **Disadvantage**: Learning Curve. SurrealQL is powerful but distinct from standard SQL.

### Value Contribution: Contextual Scoping

It enables "Contextual Scoping." By using graph relations, CodaCite can instantly narrow down its search space based on user-defined "Notebooks," providing a personalized and focused AI experience.

---

## 4. Structural Context Chunking

### Rationale: Semantic Integrity

Standard "fixed-size" chunking (e.g., every 500 characters) often cuts through sentences or headers, losing the document's original meaning. We chose **Structural Context Chunking** to preserve the hierarchy of the information.

### Mechanism: Layout Recovery

The ingestion pipeline uses `Docling` to recover the document layout. Our chunker then:

1. **Identifies Headers**: Extracts the semantic hierarchy (H1 -> H2 -> H3).

2. **Injects Context**: Pre-pends the relevant header trail to every chunk so the AI knows which section the text belongs to.

3. **Tracks Offsets**: Stores the exact start and end characters for precise UI highlighting.

### Trade-offs: Accuracy vs. Performance

* **Advantage**: Perfect Citations. The AI can point to the exact paragraph and page because the offsets are preserved.

* **Advantage**: Better Reasoning. The LLM understands that a paragraph under "Risks" has a different meaning than one under "Benefits."

* **Disadvantage**: High Ingestion Cost. Requires layout analysis (OCR/PDF parsing), which is more CPU/GPU intensive.

### Value Contribution: Ground Truth

This is the "Ground Truth" pillar. It ensures that every response generated by CodaCite is anchored to a specific, identifiable piece of structural evidence.

---

## 5. Recursive Map-Reduce Summarization

### Rationale: Infinite Context

Standard summarization fails when documents are hundreds of pages long. The context window of even the most powerful local models is limited. We implemented **Recursive Map-Reduce Summarization** to generate high-fidelity overviews regardless of document size.

### Mechanism: Hierarchical Synthesis

The pipeline follows a multi-stage reduction:

1. **Map Step**: Each individual chunk is summarized in isolation to capture local details.

2. **Reduce Step**: These local summaries are grouped and summarized again.

3. **Recursive Iteration**: This process repeats until a single, cohesive global summary is produced.

### Trade-offs: Depth vs. Performance

* **Advantage**: Scalability. Can summarize a 1,000-page document using a 4k context window model.

* **Advantage**: Information Density. Preserves key themes from all parts of the document, not just the beginning.

* **Disadvantage**: Accumulative Hallucination. Errors in early "Map" steps can be magnified in later "Reduce" stages.

* **Disadvantage**: Costly. Requires significantly more LLM tokens than single-pass summarization.

### Value Contribution: Global Understanding

It enables "Global Understanding." Users can instantly grasp the core themes of a massive document set without reading every page, providing a powerful entry point for deeper investigation.

---

## 6. Coreference Resolution

### Rationale: Contextual Clarity

Text is full of ambiguous pronouns ("he," "it," "they") and elliptical references. When chunks are processed independently, the AI often loses track of who or what is being discussed, leading to fragmented reasoning. **Coreference Resolution** resolves these references to their explicit entities.

### Mechanism: Neural Mention Linking

During pre-ingestion:

1. **Mention Detection**: Identifies noun phrases and pronouns.

2. **Cluster Assignment**: Groups references (e.g., "Apple," "the company," "it") into a single identity cluster.

3. **Reference Replacement**: Replaces ambiguous terms with the canonical entity name (where appropriate) to enhance semantic search accuracy.

### Trade-offs: Precision vs. Computational Overhead

* **Advantage**: Improved Semantic Search. Ensures that searching for "Apple" finds segments referring to it as "it."

* **Advantage**: Enhanced Reasoning. The LLM receives clear, explicit context rather than ambiguous pointers.

* **Disadvantage**: High Processing Latency. Coreference models are typically sequential and resource-heavy.

* **Disadvantage**: Risk of Oversimplification. Aggressive replacement can sometimes change the nuances of formal or legalistic prose.

### Value Contribution: Conceptual Integrity

It ensures "Conceptual Integrity." By stripping away ambiguity, CodaCite provides the LLM with unambiguous information, preventing common errors where the model fails to link related statements across large bodies of text.

---

## 7. Hybrid Retrieval (BGE-M3)

### Rationale: Multi-Vector Precision

Pure semantic search (vector search) can struggle with acronyms, specific product IDs, or rare terminology. We implemented **Hybrid Retrieval** using the BGE-M3 model to combine the strengths of dense vector embeddings and sparse lexical keywords.

### Mechanism: BGE-M3 & BM25

The system executes two parallel searches for every query:

1. **Dense Retrieval**: Uses a 1024-dimensional vector to find conceptually similar chunks.

2. **Sparse Retrieval (BM25)**: Uses token-frequency mapping (lexical) to find exact keyword matches.

The results are then combined using **Reciprocal Rank Fusion (RRF)** to ensure the most relevant candidates from both methods rise to the top.

### Trade-offs: Coverage vs. Overhead

* **Advantage**: High Recall. Finds results even when keywords don't match exactly, while still catching specific "needle-in-a-haystack" terms.

* **Advantage**: Acronym Resilience. Handles industry-specific jargon that might not be well-represented in standard embedding spaces.

* **Disadvantage**: Compute Intensity. Running dual-search paths increases the initial retrieval latency.

* **Disadvantage**: Storage Cost. Storing both dense vectors and sparse indices increases the database footprint.

### Value Contribution: Robust Discovery

This ensures "Zero-Failure Discovery." Whether a user searches for a broad concept or a specific serial number, CodaCite provides the correct context, significantly reducing the "I can't find that document" frustration.

---

## 8. Entity Resolution & Semantic Blocking

### Rationale: Graph Clarity

When extracting entities (People, Projects, Organizations) from hundreds of chunks, the same entity often appears in different forms (e.g., "Google," "Google Inc," "Google LLC"). Without **Entity Resolution**, the Knowledge Graph becomes fragmented and unusable.

### Mechanism: Semantic Blocking & Cross-Encoding

The system uses a two-stage merge pipeline:

1. **Semantic Blocking**: Groups entity candidates based on name similarity and vector closeness to reduce the number of comparisons.

2. **Cross-Encoder Verification**: A specialized model (e.g., `BGE-Reranker`) compares pairs within a block to determine if they are truly the same entity.

3. **Graph Merging**: If verified, the system merges the nodes in SurrealDB, preserving all original relationships.

### Trade-offs: Deduplication vs. Resolution Logic

* **Advantage**: Clean Knowledge Graphs. Prevents duplicate nodes from cluttering the visualization and polluting the retrieval context.

* **Advantage**: Insight Aggregation. Relationships from different documents are correctly attributed to a single, authoritative entity node.

* **Disadvantage**: Processing Time. Resolution is a computationally expensive post-ingestion step.

* **Disadvantage**: Risk of False Merges. Over-aggressive resolution can merge distinct entities with similar names.

### Value Contribution: Structured Synthesis

It transforms "Isolated Facts" into "Unified Knowledge." By resolving entities, CodaCite allows users to see the complete web of relationships across their entire document library, rather than just disconnected mentions.

---

## 9. Hardware-Aware Dynamic Routing

### Rationale: Local-First Resilience

CodaCite is a "local-first" tool. Users might run it on a high-end workstation with a GPU or a laptop with just a CPU. We implemented **Dynamic VRAM Allocation** to ensure performance without crashing.

### Mechanism: Runtime Probing

At runtime, the system probes the hardware (using `torch.cuda` or `torch.backends.mps`):

* **GPU Path**: If > 1.5GB VRAM is free, heavy tasks like PDF extraction (OCR) are routed to the GPU.

* **CPU Fallback**: If VRAM is constrained, the system gracefully shifts to a optimized CPU-only model (using `OpenVINO`).

### Trade-offs: Versatility vs. Complexity

* **Advantage**: Resilience. The system won't crash with Out-Of-Memory (OOM) errors on lower-end hardware.

* **Advantage**: Efficiency. Users with powerful hardware get a 10x speedup in ingestion.

* **Disadvantage**: Complex Maintenance. We must maintain multiple codepaths and ensure consistency between CPU and GPU outputs.

### Value Contribution: Democratized AI

It makes CodaCite "Democratized AI." It works for the developer with a desktop and the analyst with a MacBook, providing the best possible experience based on available resources.

---

## 10. Real-Time Streaming (SSE)

### Rationale: Perceived Latency

Generating high-quality, grounded answers with local LLMs can take several seconds. Without real-time feedback, the UI feels "stuck." We use **Server-Sent Events (SSE)** to provide an interactive, low-latency experience.

### Mechanism: Generator-to-Frontend Pushing

The backend uses FastAPI's `StreamingResponse` to push data chunks to the browser as they are generated:

1. **Status Events**: "Thinking...", "Retrieving...", "Synthesizing...".

2. **Token Streaming**: Pushes individual words or characters to the UI for "typewriter" style rendering.

3. **Citation Payloads**: Sends verified citation metadata once the generation is complete.

### Trade-offs: Interaction vs. Complexity

* **Advantage**: Instant Feedback. The user sees the system working immediately, significantly reducing perceived wait time.

* **Advantage**: Dynamic Updates. We can update citation badges or warning flags in real-time as the model reflects on its output.

* **Disadvantage**: Connection Stability. Maintaining long-lived SSE connections can be challenging in some network environments.

* **Disadvantage**: State Management. The frontend must handle partial, potentially out-of-order streams and reconstruct the final message.

### Value Contribution: Interactive Trust

It fosters "Interactive Trust." By showing the "thought process" and streaming the answer, CodaCite feels like a responsive assistant rather than a slow, black-box processing engine.

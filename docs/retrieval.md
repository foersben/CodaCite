# Retrieval

Retrieval within CodaCite transcends conventional methodologies through an advanced hybrid mechanism, commonly designated as Graph-based Retrieval-Augmented Generation. This retrieval pipeline is ingeniously constructed to overcome the inherent limitations of simple vector search, which often fails to capture the broader, interconnected context of a nuanced query.

## Notebook-Scoped Search

A major architectural pillar of CodaCite is the ability to perform **Notebook-Scoped Retrieval**. Instead of searching across the entire global database, the system allows users to select specific "Notebooks" to define the active context.

When a query is issued, the retrieval engine applies a graph-based filter:

1. **Scope Definition**: The user provides a set of `notebook_ids`.
2. **Graph Filtering**: The system restricts both vector search and graph traversal to only those chunks and entities that are reachable through `belongs_to` relationships with the selected notebooks.
3. **Responsive Recalculation**: As users toggle notebooks in the UI, the active context is instantly updated, allowing for highly specific and relevant AI interactions.

The retrieval pipeline is orchestrated into five distinct stages, ensuring a robust and verifiable grounding for the generative response:

1. **Stage 1: Multi-Modal Query Processing**: The system calculates the query's dense vector embedding using the **BGE-M3** model. Simultaneously, it extracts named entities and conceptual keywords using a lightweight local extractor (e.g., **GLiNER**) to seed the graph traversal.
2. **Stage 2: Notebook-Scoped Vector Search**: Interrogates the **SurrealDB HNSW** index. Crucially, the search is scoped using `SurrealQL`'s graph filtering to ensure that only chunks belonging to the selected `Notebook` are considered.
3. **Stage 3: Multi-Hop Graph Traversal**: Starting from the seed nodes identified in Stage 1, the engine executes a **Breadth-First Search (BFS)** up to 2-3 hops. This identifies "hidden" context—relationships that are semantically distant in vector space but logically connected in the graph (e.g., "Person A" -> "works at" -> "Company B").
4. **Stage 4: Context Synthesis & Aggregation**: Chunks, entity descriptions, and relationship triples are aggregated into a unified context packet. The system ensures that the source lineage (RecordIDs) is preserved for precise citation generation.
5. **Stage 5: Cross-Encoder Reranking (Optional)**: If enabled, a specialized reranker (e.g., `BGE-Reranker-v2-m3`) evaluates the query against the combined context to prune irrelevant data and prioritize the most significant evidence for the LLM.

The culmination of this hybrid retrieval process merges the deep semantic chunks identified via vector search with the structured relational context harvested from the graph traversal, providing a comprehensive "world model" for the query.

```mermaid
graph TD
    QUERY[User Query] --> SCOPE[Define Notebook Scope]
    SCOPE --> EMBED[Calculate Vector Embedding]
    SCOPE --> LINK[Entity Linking & Keyword Extraction]

    EMBED --> VECTOR[Filtered Vector Search]
    VECTOR --> CHUNKS[Retrieve Scoped Chunks]

    LINK --> SEEDS[Identify Scoped Seed Nodes]
    SEEDS --> TRAVERSE[Multi-hop Graph Traversal]
    TRAVERSE --> CONTEXT[Harvest Relational Context]

    CHUNKS --> MERGE[Synthesize Dataset]
    CONTEXT --> MERGE

    MERGE --> RERANK[Rerank & Filter]
    RERANK --> GEN[Generative Language Model Prompt]
```

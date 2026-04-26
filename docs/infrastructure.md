# Infrastructure

At the core of this expansive architecture lies the foundational role of SurrealDB, selected explicitly for its unparalleled versatility as a primary storage engine. Embracing a multi-model paradigm, the database seamlessly and simultaneously acts as a highly scalable document store for the meticulously processed text chunks and as a native graph database capable of traversing complex relationships. This dual capability eliminates the impedance mismatch typically encountered when synchronizing separate relational and graph systems. By unifying the storage layer, the application ensures that the textual representation of a chunk and its corresponding graph topology are maintained in absolute synchrony, guaranteeing data integrity and transactional consistency across the entire intelligence lifecycle.

The infrastructure heavily leverages the implementation of Hierarchical Navigable Small World indices, specifically configured as MTREE structures within the SurrealDB environment. These advanced indices facilitate highly optimized cosine distance searches across vast high-dimensional vector spaces. When the application generates dense embeddings representing the semantic meaning of text chunks or entity descriptions, these vectors are inserted into the index, allowing the system to achieve extraordinary performance when retrieving semantically similar content. This mathematical optimization is crucial for anchoring the system's massive throughput capabilities, ensuring that complex similarity calculations do not become a computational bottleneck during real-time user interactions or concurrent batch processing.

Beyond the real-time operational demands, the infrastructure supports continuous, asynchronous refinement processes such as graph enhancement and community detection. Operating in the background, these refinement pipelines analyze the ever-expanding knowledge graph to deduce macro-level structural patterns. By employing advanced community detection algorithms, such as the Louvain method via the NetworkX library, the system algorithmically identifies dense clusters and thematic groupings of nodes that share intense relational gravity. The discovery of these underlying communities enables the application to transcend micro-level facts, allowing the generative language model to synthesize holistic insights and panoramic overviews of the entire knowledge base based on the structural topology of the data itself.

## Data Consistency and Type Harmonization

To maintain the rigorous standards of the internal Pydantic domain models, the infrastructure implements a critical type-harmonization layer between the application and the SurrealDB storage engine. SurrealDB utilizes specialized `RecordID` objects for referencing data, which inherently contain database-specific prefixes (e.g., `chunk:`, `entity:`). To ensure seamless compatibility with strict-mode validation, the application's storage adapters (`SurrealDocumentStore` and `SurrealGraphStore`) automatically sanitize these identifiers during retrieval. This process casts complex database types into pure string representations and strips prefixes, ensuring that the domain logic remains isolated from database-level implementation details while preserving the integrity of the unique identifier across the entire system.

## Adaptive Resource Allocation

The infrastructure is designed for high resilience in diverse hardware environments through an adaptive resource allocation strategy. Recognizing the significant memory overhead of modern machine learning models, the application strictly respects a system-level `DEVICE` configuration. Components such as the `HuggingFaceEmbedder`, `GLiNERFallbackExtractor`, and `FastCorefResolver` are instrumented to prioritize this setting, enabling a graceful fallback to CPU-based inference when GPU resources are constrained or unavailable. This strategy prevents critical `CUDA Out of Memory` failures, ensuring that the document intelligence pipeline remains operational even on commodity hardware or within shared containerized environments.

## Secure Secret Management

To eliminate the risks associated with hardcoded credentials and manual environment variable management, the infrastructure integrates directly with the system's **Secret Service** (libsecret). This is particularly optimized for development environments using **KeePassXC**.

The application configuration layer automatically attempts to resolve the Gemini API key from the local secret store using the following parameters:

- **Label/Title**: `Gemini_API`

This integration utilizes the `secretstorage` library to communicate directly with the D-Bus secret service, ensuring that sensitive keys remain encrypted at rest and are only accessed in-memory during application startup. If a key is explicitly provided via the `GEMINI_API_KEY` environment variable, it will always take precedence over the secret store, allowing for flexible deployment across both local and production environments.

## Automated Maintenance and Index Health

To ensure long-term performance and data integrity within the SurrealDB HNSW vector index, the infrastructure implements **Automated Maintenance Loops**. Vector indices, particularly those using HNSW, can suffer from performance degradation due to "tombstones" (logical deletions that aren't immediately purged from the index structure).

The `SurrealDocumentStore` tracks a global deletion counter. Every 5 document deletions, the system automatically triggers a background maintenance routine:

1. **Index Rebuilding**: The `REBUILD INDEX chunk_embedding_idx` command is issued to SurrealDB.
2. **Tombstone Purging**: Rebuilding the index physically removes deleted vectors and re-optimizes the navigable graph structure.
3. **Graph Integrity**: Relational edges (like `belongs_to` and `mentions`) are transactionally removed alongside their parent nodes to prevent "dangling relations."

This self-healing mechanism ensures that semantic search remains fast and accurate without requiring manual intervention from system administrators.

# Data Pipeline

The journey of any given piece of unstructured data commences at the ingestion and preprocessing lifecycle, a critical phase dedicated to standardizing chaotic information into a precise analytical format. The document loader acts as the initial conduit, parsing disparate file types, including complex portable document formats and markdown structures, into a unified internal representation. Subsequently, the text preprocessor undertakes rigorous normalization routines, resolving anomalous Unicode characters and systematically compressing errant whitespace. This foundational sanitization ensures that the underlying semantic integrity of the document is preserved against the idiosyncrasies of human formatting. Before the standardized text is subjected to the chunking strategy, it undergoes a vital linguistic transformation through coreference resolution. Powered by specialized libraries such as **`fastcoref`**, this step analyzes the grammatical structure to unify disparate pronoun and noun mentions back to their primary entities. By resolving these references across the continuum of the text, the system guarantees that the contextual weight of an entity is completely captured, preventing semantic fragmentation when the document is ultimately partitioned into analytical chunks.

## The 8-Phase Ingestion Pipeline

CodaCite orchestrates a sophisticated, asynchronous ingestion choreography:

1.  **Phase 1: Loading & Preprocessing**: File validation and text extraction (PDF/Text) with NFKC normalization.
2.  **Phase 2: Coreference Resolution**: Uses `fastcoref` to normalize linguistic references across the document.
3.  **Phase 3: Recursive Chunking**: Leverages **LangChain's** `RecursiveCharacterTextSplitter` to partition text into overlapping fragments.
4.  **Phase 4: Document Persistence**: Commits raw text chunks and establishes notebook relations in **SurrealDB**.
5.  **Phase 5: Vectorization (Embedding)**: Generates 1024D vectors using the **BGE-M3** transformer model.
6.  **Phase 6: Knowledge Extraction**: Utilizes **Google Gemini** (or **GLiNER** fallback) to identify entity nodes and relationship edges.
7.  **Phase 7: Entity Resolution**: Resolves extracted nodes against the global graph using Jaro-Winkler similarity and vector distance.
8.  **Phase 8: Finalization**: Updates the document status to `active` and rebuilds vector indices.

## Model Interaction and Tokenization

Once the preprocessing algorithms have refined the text, the resulting semantic chunks are propelled into the knowledge graph extraction and embedding phases. It is important to note the transition from "clean text" to "model tokens" during these stages:

1.  **Tokenization**: Before an AI model (BGE-M3 or Gemini) can process a chunk, it is converted into a sequence of numerical tokens using the model's specific tokenizer (e.g., SentencePiece for BGE-M3, or the Gemini-specific tokenizer). Tokenization partitions the text into sub-word units (tokens) that the model's neural layers can process mathematically.
2.  **Vectorization**: The embedding model (Phase 5) processes these tokens through its transformer blocks to generate a high-dimensional vector (1024D) representing the semantic "gravity" or meaning of the chunk.
3.  **Inference & Extraction**: The extraction model (Phase 6) utilizes the tokenized context window to perform structured reasoning. It identifies actors and relationships based on the provided Pydantic schemas, ensuring the output is perfectly mapped to the application's domain models.

This transition ensures that the "messy" human text from Phase 1 is transformed into a mathematically precise representation that the AI system can effectively reason about.

## Observability and Instrumentation

Recognizing the inherent ambiguity and variability in natural language, the system employs advanced resolution techniques to prevent the proliferation of duplicate entities. A specialized resolver component, utilizing algorithms like the **Jaro-Winkler distance**, calculates complex string similarities and evaluates corresponding high-dimensional vector embeddings to determine if newly extracted nodes refer to existing concepts within the graph. This meticulous reconciliation process is imperative to maintaining a coherent and singular source of truth within the database. By continuously collapsing synonymous entities and merging their relational edges, the data pipeline ensures that the knowledge graph matures into a dense, highly connected web of intelligence rather than a fragmented collection of redundant, disconnected nodes.

## Observability and Instrumentation

To facilitate granular monitoring of the complex ingestion lifecycle, the data pipeline is instrumented with a comprehensive observability framework. Every document transition—from initial upload and chunking to coreference resolution and final graph insertion—is meticulously tracked using structured `[INGEST]` logging tags. This telemetry allows operators to pinpoint bottlenecks or failures in real-time, providing immediate visibility into the state of any given document within the processing queue. By surface-leveling these internal metrics, the system ensures that the document intelligence pipeline remains transparent and maintainable, even as the volume of ingested data scales into the millions of entities.

```mermaid
graph LR
    DOC[Raw Document] --> INGEST[Ingestion & Standardization]
    INGEST --> PREPROC[Text Preprocessing]
    PREPROC --> CORE[Coreference Resolution]
    CORE --> CHUNK[Semantic Chunking]
    CHUNK --> EXTRACT[Knowledge Extraction]
    EXTRACT --> RESOLVE[Entity Resolution]
    RESOLVE --> STORE[(Knowledge Graph Store)]
```

"""SurrealDB schema definitions."""


def get_schema_queries(embedding_dim: int = 1024) -> list[str]:
    """Return SurrealQL queries to initialize the production database schema.

    The schema is designed for a hybrid GraphRAG approach, combining document
    metadata, vector chunks, and entity relationships.

    Layers:
        1.  **Organizational Layer**: `notebook` (folders) and `document` (files).
            Linked via `belongs_to` edges.
        2.  **Semantic Chunk Layer**: `chunk` table storing vectorized text fragments.
            Linked to documents via `contains` edges.
        3.  **Knowledge Graph Layer**: `entity` nodes (extracted concepts) and
            `relation` edges (extracted semantic links). Entities are linked
            back to their source text via `extracted_from` edges.
        4.  **Vector Indices**: HNSW indices on both `chunk.embedding` and
            `entity.description_embedding` for semantic retrieval.
        5.  **Full-Text Indices**: BM25 index on `chunk.text` via the `standard`
            analyzer for keyword-based retrieval using the ``@1@`` operator.

    Args:
        embedding_dim: Dimension of the vector embeddings (default: 1024 for BGE).

    Returns:
        A list of SurrealQL string blocks to be executed in sequence.
    """
    # 1. Notebooks and Documents
    base_schema = """
    -- Notebook Container
    DEFINE TABLE notebook SCHEMAFULL;
    DEFINE FIELD name ON notebook TYPE string;
    DEFINE FIELD created_at ON notebook TYPE datetime DEFAULT time::now();

    -- Document Metadata (Blob Storage Pointer)
    DEFINE TABLE document SCHEMAFULL;
    DEFINE FIELD filename ON document TYPE string;
    DEFINE FIELD file_path ON document TYPE string; -- Path in /app/uploads
    DEFINE FIELD status ON document TYPE string ASSERT $value IN ['processing', 'active', 'failed'];
    DEFINE FIELD metadata ON document TYPE object;
    DEFINE FIELD created_at ON document TYPE datetime DEFAULT time::now();

    -- Relationships: Document -> belongs_to -> Notebook
    DEFINE TABLE belongs_to SCHEMAFULL TYPE RELATION FROM document TO notebook;

    -- Maintenance: Deleting a document should delete its relationship edges
    DEFINE EVENT delete_doc_edges ON TABLE document WHEN $event = "DELETE" THEN {{
        DELETE belongs_to WHERE in = $before.id;
        DELETE contains WHERE in = $before.id;
    }};
    """

    # 2. Chunks, Vector Index, and Full-Text Search Index
    chunk_schema = f"""
    DEFINE TABLE chunk SCHEMAFULL;
    DEFINE FIELD document_id ON chunk TYPE string;
    DEFINE FIELD text ON chunk TYPE string;
    DEFINE FIELD index ON chunk TYPE int;
    DEFINE FIELD embedding ON chunk TYPE array<float>;

    -- Relationships: Document -> contains -> Chunk
    DEFINE TABLE contains SCHEMAFULL TYPE RELATION FROM document TO chunk;

    -- Analyzer for BM25 full-text search: lowercases and applies english stemming
    DEFINE ANALYZER standard TOKENIZERS class FILTERS lowercase, snowball(english);

    -- BM25 Full-Text Search Index on chunk text (enables @1@ operator)
    DEFINE INDEX chunk_text_idx ON TABLE chunk FIELDS text SEARCH ANALYZER standard BM25(1.2, 0.75) HIGHLIGHTS;

    -- HNSW Index for semantic vector search
    DEFINE INDEX chunk_embedding_idx ON TABLE chunk FIELDS embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;

    -- Maintenance: Deleting a chunk should delete its extraction edges
    DEFINE EVENT delete_chunk_edges ON TABLE chunk WHEN $event = "DELETE" THEN {{
        DELETE extracted_from WHERE out = $before.id;
    }};
    """

    # 3. Entity Nodes and Graph Relationships
    graph_schema = f"""
    DEFINE TABLE entity SCHEMAFULL;
    DEFINE FIELD label ON entity TYPE string;
    DEFINE FIELD name ON entity TYPE string;
    DEFINE FIELD description ON entity TYPE option<string>;
    DEFINE FIELD description_embedding ON entity TYPE option<array<float>>;

    -- Relationship: Entity -> extracted_from -> Chunk
    DEFINE TABLE extracted_from SCHEMAFULL TYPE RELATION FROM entity TO chunk;

    -- Entity Embedding Index
    DEFINE INDEX entity_embedding_idx ON TABLE entity FIELDS description_embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;

    -- Semantic Relationship between Entities
    DEFINE TABLE relation SCHEMAFULL TYPE RELATION FROM entity TO entity;
    DEFINE FIELD relation ON relation TYPE string;
    DEFINE FIELD description ON relation TYPE option<string>;
    DEFINE FIELD weight ON relation TYPE float DEFAULT 1.0;
    """

    # 4. Maintenance Counts
    maintenance_schema = """
    DEFINE TABLE maintenance SCHEMAFULL;
    DEFINE FIELD count ON maintenance TYPE int DEFAULT 0;
    """

    return [base_schema, chunk_schema, graph_schema, maintenance_schema]

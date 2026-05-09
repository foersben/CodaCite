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
    base_queries = [
        "DEFINE TABLE OVERWRITE notebook SCHEMAFULL;",
        "DEFINE FIELD OVERWRITE title ON notebook TYPE string;",
        "DEFINE FIELD OVERWRITE description ON notebook TYPE option<string>;",
        "DEFINE FIELD OVERWRITE created_at ON notebook TYPE option<string>;",
        "DEFINE TABLE OVERWRITE document SCHEMAFULL;",
        "DEFINE FIELD OVERWRITE filename ON document TYPE string;",
        "DEFINE FIELD OVERWRITE file_path ON document TYPE string;",
        "DEFINE FIELD OVERWRITE status ON document TYPE string ASSERT $value IN ['processing', 'active', 'failed'];",
        "DEFINE FIELD OVERWRITE metadata ON document TYPE object;",
        "DEFINE FIELD OVERWRITE created_at ON document TYPE datetime DEFAULT time::now();",
        "DEFINE TABLE OVERWRITE belongs_to SCHEMAFULL TYPE RELATION FROM document TO notebook;",
        """
        DEFINE EVENT delete_doc_edges ON TABLE document WHEN $event = "DELETE" THEN {
            DELETE belongs_to WHERE in = $before.id;
            DELETE contains WHERE in = $before.id;
        };
        """,
    ]

    # 2. Chunks and Search Indices
    chunk_queries = [
        "DEFINE TABLE OVERWRITE chunk SCHEMAFULL;",
        "DEFINE FIELD OVERWRITE document_id ON chunk TYPE string;",
        "DEFINE FIELD OVERWRITE text ON chunk TYPE string;",
        "DEFINE FIELD OVERWRITE index ON chunk TYPE int;",
        "DEFINE FIELD OVERWRITE start_char ON chunk TYPE int DEFAULT 0;",
        "DEFINE FIELD OVERWRITE end_char ON chunk TYPE int DEFAULT 0;",
        "DEFINE FIELD OVERWRITE embedding ON chunk TYPE array<float>;",
        "DEFINE TABLE OVERWRITE contains SCHEMAFULL TYPE RELATION FROM document TO chunk;",
        "DEFINE ANALYZER OVERWRITE standard TOKENIZERS class FILTERS lowercase, snowball(english);",
        "DEFINE INDEX OVERWRITE chunk_text_idx ON TABLE chunk FIELDS text FULLTEXT ANALYZER standard BM25(1.2, 0.75) HIGHLIGHTS;",
        f"DEFINE INDEX OVERWRITE chunk_embedding_idx ON TABLE chunk FIELDS embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;",
        """
        DEFINE EVENT delete_chunk_edges ON TABLE chunk WHEN $event = "DELETE" THEN {
            DELETE extracted_from WHERE out = $before.id;
        };
        """,
    ]

    # 3. Entity Nodes and Graph Relationships
    graph_queries = [
        "DEFINE TABLE OVERWRITE entity SCHEMAFULL;",
        "DEFINE FIELD OVERWRITE label ON entity TYPE string;",
        "DEFINE FIELD OVERWRITE name ON entity TYPE string;",
        "DEFINE FIELD OVERWRITE description ON entity TYPE option<string>;",
        "DEFINE FIELD OVERWRITE description_embedding ON entity TYPE option<array<float>>;",
        "DEFINE TABLE OVERWRITE extracted_from SCHEMAFULL TYPE RELATION FROM entity TO chunk;",
        f"DEFINE INDEX OVERWRITE entity_embedding_idx ON TABLE entity FIELDS description_embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;",
        "DEFINE TABLE OVERWRITE relation SCHEMAFULL TYPE RELATION FROM entity TO entity;",
        "DEFINE FIELD OVERWRITE relation ON relation TYPE string;",
        "DEFINE FIELD OVERWRITE description ON relation TYPE option<string>;",
        "DEFINE FIELD OVERWRITE weight ON relation TYPE float DEFAULT 1.0;",
    ]

    # 4. Maintenance Counts
    maintenance_queries = [
        "DEFINE TABLE OVERWRITE maintenance SCHEMAFULL;",
        "DEFINE FIELD OVERWRITE count ON maintenance TYPE int DEFAULT 0;",
    ]

    return base_queries + chunk_queries + graph_queries + maintenance_queries

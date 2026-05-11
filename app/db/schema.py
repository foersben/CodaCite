"""SurrealDB schema definitions for GraphRAG.

This module defines the structural blueprint of the CodaCite database. It
utilizes SurrealQL's schema-full capabilities to enforce data integrity across
the organizational, semantic, and knowledge graph layers of the system.
"""


def get_schema_queries(embedding_dim: int = 1024) -> list[str]:
    """Generate the full suite of SurrealQL queries to initialize the schema.

    The schema is architected to support a multi-layered hybrid retrieval
    strategy:
    1.  **Organizational Layer**: Manages notebooks and documents using relational
        scoping via `belongs_to` edges.
    2.  **Semantic Chunk Layer**: Stores text fragments with HNSW vector indices
        and BM25 full-text indices for high-performance hybrid search.
    3.  **Knowledge Graph Layer**: Encapsulates entities and their semantic
        relationships, linked back to source chunks via `extracted_from` edges.
    4.  **Auto-Maintenance**: Employs database events to ensure referential
        integrity during deletions.

    Args:
        embedding_dim: The dimensionality of the vector embeddings, typically
            aligned with the active transformer model (e.g., 1024 for BGE-v1.5).

    Returns:
        A sequential list of SurrealQL strings used to define tables, fields,
        analyzers, events, and indices.
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
        "DEFINE FIELD OVERWRITE global_summary ON document TYPE option<string>;",
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
        "DEFINE INDEX OVERWRITE chunk_text_idx ON TABLE chunk COLUMNS text FULLTEXT ANALYZER standard BM25(1.2, 0.75) HIGHLIGHTS;",
        f"DEFINE INDEX OVERWRITE chunk_embedding_idx ON TABLE chunk COLUMNS embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;",
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
        "DEFINE INDEX OVERWRITE entity_name_idx ON TABLE entity COLUMNS name FULLTEXT ANALYZER standard BM25(1.2, 0.75) HIGHLIGHTS;",
        f"DEFINE INDEX OVERWRITE entity_embedding_idx ON TABLE entity COLUMNS description_embedding HNSW DIMENSION {embedding_dim} DIST COSINE EFC 150 M 12 TYPE F32;",
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

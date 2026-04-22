"""SurrealDB schema definitions."""


def get_schema_queries(embedding_dim: int = 1024) -> list[str]:
    """Return SurrealQL queries to initialize the database schema."""
    # 1. Chunk Table and HNSW Vector Index
    chunk_schema = f"""
    DEFINE TABLE chunk SCHEMAFULL;
    DEFINE FIELD id ON chunk TYPE string;
    DEFINE FIELD document_id ON chunk TYPE string;
    DEFINE FIELD text ON chunk TYPE string;
    DEFINE FIELD index ON chunk TYPE int;
    DEFINE FIELD embedding ON chunk TYPE array<float>;

    DEFINE INDEX chunk_embedding_idx ON chunk FIELDS embedding MTREE DIMENSION {embedding_dim} DIST COSINE;
    """

    # 2. Entity Node Table and HNSW Vector Index
    node_schema = f"""
    DEFINE TABLE entity SCHEMAFULL;
    DEFINE FIELD id ON entity TYPE string;
    DEFINE FIELD label ON entity TYPE string;
    DEFINE FIELD name ON entity TYPE string;
    DEFINE FIELD description ON entity TYPE option<string>;
    DEFINE FIELD description_embedding ON entity TYPE option<array<float>>;
    DEFINE FIELD source_chunk_ids ON entity TYPE array<string>;

    DEFINE INDEX entity_embedding_idx ON entity FIELDS description_embedding MTREE DIMENSION {embedding_dim} DIST COSINE;
    """

    # 3. Community Summary Table
    community_schema = """
    DEFINE TABLE community SCHEMAFULL;
    DEFINE FIELD id ON community TYPE string;
    DEFINE FIELD summary ON community TYPE string;
    DEFINE FIELD node_ids ON community TYPE array<string>;
    """

    # Relationships are handled dynamically by SurrealDB edge tables,
    # but we can enforce the relation schema:
    relation_schema = """
    DEFINE TABLE relation SCHEMAFULL TYPE RELATION FROM entity TO entity;
    DEFINE FIELD relation ON relation TYPE string;
    DEFINE FIELD description ON relation TYPE option<string>;
    DEFINE FIELD source_chunk_ids ON relation TYPE array<string>;
    DEFINE FIELD weight ON relation TYPE float;
    """

    return [chunk_schema, node_schema, community_schema, relation_schema]

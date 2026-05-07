"""Use case for extracting knowledge graphs from text chunks.

This module coordinates the extraction of entities and relationships from
pre-processed document chunks, handles entity resolution, and persists the
resulting graph structure.
"""

import logging
from typing import TYPE_CHECKING

from app.core.interfaces import Embedder, EntityExtractor, EntityResolver, GraphStore
from app.models.models import Chunk, Edge, Node

if TYPE_CHECKING:
    from .llm_relator import LLMRelator

logger = logging.getLogger(__name__)


class GraphExtractionUseCase:
    """Coordinates the extraction and resolution of Knowledge Graphs.

    This use case processes refined text chunks to build a semantic graph
    structure. It handles the iterative extraction, entity resolution,
    vectorization of concepts, and final persistence.

    Extraction Pipeline:
        1.  **Iterative Extraction**: Calls `EntityExtractor` for each chunk.
        2.  **Source Attribution**: Tags nodes/edges with source chunk IDs for
            citation traceability.
        3.  **Global Resolution**: Uses `EntityResolver` to merge new nodes
            with existing entities in the `GraphStore`.
        4.  **Concept Vectorization**: Generates embeddings for entity
            descriptions to enable conceptual retrieval.
        5.  **Relation Normalization**: Standardizes relationship labels
            (e.g., "WORKS_AT" -> "WORKS_FOR").
        6.  **Persistence**: Commits the resulting subgraph to the database.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        relator: "LLMRelator",
        resolver: EntityResolver,
        graph_store: GraphStore,
        embedder: Embedder,
    ) -> None:
        """Initialize the extraction use case with required infrastructure.

        Args:
            extractor: Logic for identifying nodes in text (Stage 1).
            relator: Logic for mapping relationships between nodes (Stage 2).
            resolver: Logic for entity deduplication and merging.
            graph_store: Persistent storage for graph data.
            embedder: Transformer model for vectorizing concepts.
        """
        self.extractor = extractor
        self.relator = relator
        self.resolver = resolver
        self.graph_store = graph_store
        self.embedder = embedder

    async def execute(self, chunks: list[Chunk]) -> tuple[list[Node], list[Edge]]:
        """Execute the two-stage hybrid graph extraction process.

        Args:
            chunks: A list of text chunks to process.

        Returns:
            A tuple containing the final list of resolved Nodes and extracted Edges.
        """
        logger.info("[EXTRACTION] Starting hybrid extraction for %d chunks", len(chunks))

        # 0. Awakening infrastructure
        await self.extractor.ensure_loaded()

        # STAGE 1: Entity Spotting (per chunk)
        raw_nodes_per_chunk: list[list[Node]] = []
        all_raw_nodes: list[Node] = []

        for i, chunk in enumerate(chunks):
            logger.debug("[EXTRACTION] Spotting entities in chunk %d/%d", i + 1, len(chunks))
            nodes, _ = await self.extractor.extract(chunk.text)

            # Tag with source chunk
            for n in nodes:
                if chunk.id not in n.source_chunk_ids:
                    n.source_chunk_ids.append(chunk.id)

            raw_nodes_per_chunk.append(nodes)
            all_raw_nodes.extend(nodes)

        if not all_raw_nodes:
            logger.warning("[EXTRACTION] No entities spotted in provided chunks.")
            return [], []

        # GLOBAL RESOLUTION (Jaro-Winkler)
        logger.info("[EXTRACTION] Resolving %d raw entities", len(all_raw_nodes))
        existing_nodes = await self.graph_store.get_all_nodes()
        resolved_nodes_list = await self.resolver.resolve_entities(all_raw_nodes, existing_nodes)

        # Create mapping of raw node ID to its resolved Node
        # Since resolver returns list in same order as all_raw_nodes
        raw_to_resolved: dict[str, Node] = {}
        for raw, resolved in zip(all_raw_nodes, resolved_nodes_list, strict=True):
            raw_to_resolved[raw.id] = resolved

        # Deduplicate final nodes for persistence
        unique_nodes_dict: dict[str, Node] = {}
        for n in resolved_nodes_list:
            if n.id not in unique_nodes_dict:
                unique_nodes_dict[n.id] = n
            else:
                # Merge source attributions
                unique_nodes_dict[n.id].source_chunk_ids = list(
                    set(unique_nodes_dict[n.id].source_chunk_ids + n.source_chunk_ids)
                )

        final_nodes = list(unique_nodes_dict.values())
        logger.info("[EXTRACTION] Resolved into %d unique canonical entities", len(final_nodes))

        # STAGE 2: Relationship Mapping (per chunk)
        all_edges: list[Edge] = []
        for i, chunk in enumerate(chunks):
            # Get the canonical names of entities found in THIS chunk
            chunk_raw_nodes = raw_nodes_per_chunk[i]
            chunk_resolved_names = list({raw_to_resolved[n.id].name for n in chunk_raw_nodes})

            if len(chunk_resolved_names) < 2:
                continue

            logger.debug("[EXTRACTION] Mapping relationships for chunk %d/%d", i + 1, len(chunks))
            chunk_edges = await self.relator.extract_relationships(chunk.text, chunk_resolved_names)

            # Tag edges with source chunk
            for e in chunk_edges:
                if chunk.id not in e.source_chunk_ids:
                    e.source_chunk_ids.append(chunk.id)
                all_edges.append(e)

        logger.info("[EXTRACTION] Mapped %d total relationships", len(all_edges))

        # 4. Generate Embeddings for Canonical Nodes
        logger.info("[EXTRACTION] Vectorizing %d canonical entities", len(final_nodes))
        for node in final_nodes:
            text_to_embed = node.description if node.description else node.name
            embedding = await self.embedder.embed(text_to_embed)
            node.description_embedding = embedding

        # 5. Normalize Relationships (Post-Processing)
        for edge in all_edges:
            edge.relation = edge.relation.upper().replace(" ", "_")

        # 6. Save to SurrealDB
        logger.info(
            "[EXTRACTION] Persisting %d nodes and %d edges to SurrealDB",
            len(final_nodes),
            len(all_edges),
        )
        await self.graph_store.save_nodes(final_nodes)
        await self.graph_store.save_edges(all_edges)

        return final_nodes, all_edges

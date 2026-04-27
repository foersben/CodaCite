"""Use case for extracting knowledge graphs from text chunks.

This module coordinates the extraction of entities and relationships from
pre-processed document chunks, handles entity resolution, and persists the
resulting graph structure.
"""

import logging

from app.domain.models import Chunk, Edge, Node
from app.domain.ports import Embedder, EntityExtractor, EntityResolver, GraphStore

logger = logging.getLogger(__name__)


class GraphExtractionUseCase:
    """Use case to extract nodes and edges from text chunks and resolve them.

    Orchestrates the extraction pipeline: extraction -> tagging -> resolution -> embedding -> storage.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        resolver: EntityResolver,
        graph_store: GraphStore,
        embedder: Embedder,
    ) -> None:
        """Initialize the extraction use case.

        Args:
            extractor: Implementation of the EntityExtractor port.
            resolver: Implementation of the EntityResolver port.
            graph_store: Implementation of the GraphStore port.
            embedder: Implementation of the Embedder port.
        """
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.embedder = embedder

    async def execute(self, chunks: list[Chunk]) -> tuple[list[Node], list[Edge]]:
        """Execute the graph extraction process for a list of chunks.

        Args:
            chunks: A list of text chunks to process.

        Returns:
            A tuple containing the final list of resolved Nodes and all extracted Edges.
        """
        logger.info("[EXTRACTION] Starting graph extraction for %d chunks", len(chunks))
        all_nodes: list[Node] = []
        all_edges: list[Edge] = []

        # 1. Extraction
        for i, chunk in enumerate(chunks):
            logger.debug("[EXTRACTION] Processing chunk %d/%d", i + 1, len(chunks))
            nodes, edges = await self.extractor.extract(chunk.text)

            # Tag with source chunk
            for n in nodes:
                if chunk.id not in n.source_chunk_ids:
                    n.source_chunk_ids.append(chunk.id)
            for e in edges:
                if chunk.id not in e.source_chunk_ids:
                    e.source_chunk_ids.append(chunk.id)

            all_nodes.extend(nodes)
            all_edges.extend(edges)

        logger.info(
            "[EXTRACTION] Extracted %d raw nodes and %d raw edges", len(all_nodes), len(all_edges)
        )

        # 2. Get existing nodes to resolve against
        logger.debug("[EXTRACTION] Fetching existing nodes for resolution")
        existing_nodes = await self.graph_store.get_all_nodes()

        # 3. Resolve Entities
        logger.info(
            "[EXTRACTION] Resolving %d entities against %d existing nodes",
            len(all_nodes),
            len(existing_nodes),
        )
        resolved_nodes = await self.resolver.resolve_entities(all_nodes, existing_nodes)

        # Deduplicate internal list (naively for now by ID)
        unique_nodes_dict: dict[str, Node] = {}
        for n in resolved_nodes:
            if n.id not in unique_nodes_dict:
                unique_nodes_dict[n.id] = n
            else:
                unique_nodes_dict[n.id].source_chunk_ids.extend(n.source_chunk_ids)

        for n in unique_nodes_dict.values():
            n.source_chunk_ids = list(set(n.source_chunk_ids))

        final_nodes = list(unique_nodes_dict.values())
        logger.info("[EXTRACTION] Resolved into %d unique entities", len(final_nodes))

        # 4. Generate Embeddings for Nodes
        logger.info("[EXTRACTION] Generating embeddings for %d entities", len(final_nodes))
        for node in final_nodes:
            # Prefer description if available for embedding
            text_to_embed = node.description if node.description else node.name
            embedding = await self.embedder.embed(text_to_embed)
            node.description_embedding = embedding

        # 5. Normalization (simple mapping example)
        for edge in all_edges:
            rel = edge.relation.upper().replace(" ", "_")
            if rel in ["IS_CEO_OF", "WORKS_AS_CEO"]:
                edge.relation = "CEO_OF"
            elif rel in ["WORKS_AT", "EMPLOYED_BY"]:
                edge.relation = "WORKS_FOR"
            else:
                # Still normalize to upper case for consistency
                edge.relation = rel

        # 6. Save to Graph Store
        logger.info(
            "[EXTRACTION] Saving %d entities and %d edges to graph store",
            len(final_nodes),
            len(all_edges),
        )
        await self.graph_store.save_nodes(final_nodes)
        await self.graph_store.save_edges(all_edges)
        logger.info("[EXTRACTION] Graph extraction and storage complete")

        return final_nodes, all_edges

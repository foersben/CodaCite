"""Use case for enhancing the knowledge graph with community detection.

This module provides the logic for detecting communities of entities within
the graph using algorithms like Louvain and summarizing them for higher-level
retrieval.
"""

import logging
import uuid
from collections.abc import Callable, Coroutine

import networkx as nx

from app.domain.models import Community
from app.domain.ports import GraphStore

logger = logging.getLogger(__name__)


class GraphEnhancementUseCase:
    """Use case to enhance the graph, e.g., via community detection.

    Processes the global graph to identify clusters of related entities and
    generates summaries for these clusters (communities).
    """

    def __init__(
        self,
        graph_store: GraphStore,
        llm_summarizer: Callable[[str], Coroutine[None, None, str]] | None = None,
    ) -> None:
        """Initialize the enhancement use case.

        Args:
            graph_store: Implementation of the GraphStore port.
            llm_summarizer: An optional async callable that takes a prompt and
                returns an LLM-generated summary string.
        """
        self.graph_store = graph_store
        self.llm_summarizer = llm_summarizer

    async def execute(self) -> None:
        """Run Louvain community detection on the entire graph and store results.

        Fetches all nodes and edges, builds a NetworkX representation,
        detects communities, and optionally summarizes them via LLM.
        """
        logger.info("[ENHANCEMENT] Starting community detection...")
        nodes = await self.graph_store.get_all_nodes()
        edges = await self.graph_store.get_all_edges()

        if not nodes or not edges:
            logger.warning("[ENHANCEMENT] Graph is empty; skipping community detection")
            return

        # Build networkx graph
        g_net = nx.Graph()
        for node in nodes:
            g_net.add_node(node.id, label=node.label, name=node.name)

        for edge in edges:
            g_net.add_edge(
                edge.source_id, edge.target_id, weight=edge.weight, relation=edge.relation
            )

        try:
            # Louvain community detection
            logger.info("[ENHANCEMENT] Running Louvain algorithm...")
            communities_generator = nx.community.louvain_communities(g_net, weight="weight")

            for i, comm in enumerate(communities_generator):
                comm_nodes = list(comm)
                logger.debug("[ENHANCEMENT] Found community %d with %d nodes", i, len(comm_nodes))

                # Naive summary (can be replaced by LLM summarize)
                summary = f"Community {i} with {len(comm_nodes)} nodes."
                if self.llm_summarizer:
                    # Collect node names for summary
                    names = [g_net.nodes[n].get("name", str(n)) for n in comm_nodes[:10]]
                    prompt = f"Summarize this community of entities: {', '.join(names)}"
                    try:
                        summary = await self.llm_summarizer(prompt)
                    except Exception as e:
                        logger.error("[ENHANCEMENT] LLM summarization failed: %s", e)

                community_model = Community(
                    id=str(uuid.uuid4()), summary=summary, node_ids=comm_nodes
                )

                await self.graph_store.save_community(community_model)

            logger.info("[ENHANCEMENT] Community detection and storage complete")

        except Exception as e:
            logger.error("[ENHANCEMENT] Community detection failed: %s", e)
            raise

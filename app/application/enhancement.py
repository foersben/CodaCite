"""Use Case for Graph Enhancements (Community Detection)."""

import uuid
from collections.abc import Callable, Coroutine

import networkx as nx

from app.domain.models import Community
from app.domain.ports import GraphStore


class GraphEnhancementUseCase:
    """Use case to enhance the graph, e.g., via community detection."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm_summarizer: Callable[[str], Coroutine[None, None, str]] | None = None,
    ) -> None:
        """Initialize the enhancement usecase.

        Args:
            graph_store: The graph store.
            llm_summarizer: An optional LLM to summarize communities.
        """
        self.graph_store = graph_store
        self.llm_summarizer = llm_summarizer

    async def execute(self) -> None:
        """Run Louvain community detection on the entire graph.

        Returns:
            None.
        """
        nodes = await self.graph_store.get_all_nodes()
        edges = await self.graph_store.get_all_edges()

        if not nodes or not edges:
            return

        # Build networkx graph
        g_net = nx.Graph()
        for node in nodes:
            g_net.add_node(node.id, label=node.label, name=node.name)

        for edge in edges:
            g_net.add_edge(edge.source_id, edge.target_id, weight=edge.weight, relation=edge.relation)

        try:
            # Louvain community detection
            communities_generator = nx.community.louvain_communities(g_net, weight="weight")

            for i, comm in enumerate(communities_generator):
                comm_nodes = list(comm)

                # Naive summary (can be replaced by LLM summarize)
                summary = f"Community {i} with {len(comm_nodes)} nodes."
                if self.llm_summarizer:
                    # Collect node names for summary
                    names = [g_net.nodes[n].get("name", str(n)) for n in comm_nodes[:10]]
                    summary = await self.llm_summarizer(
                        f"Summarize this community of entities: {names}"
                    )

                community_model = Community(
                    id=str(uuid.uuid4()), summary=summary, node_ids=comm_nodes
                )

                await self.graph_store.save_community(community_model)

        except Exception as e:
            print(f"Community detection failed: {e}")

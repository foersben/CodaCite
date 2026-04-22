"""Use Case for Advanced GraphRAG Retrieval."""

from typing import cast

from app.domain.models import Edge, Node
from app.domain.ports import DocumentStore, Embedder, GraphStore


class GraphRAGRetrievalUseCase:
    """Use case for performing hybrid GraphRAG queries."""

    def __init__(
        self,
        document_store: DocumentStore,
        graph_store: GraphStore,
        embedder: Embedder,
        entity_linker: object,
        reranker: object,
    ) -> None:
        """Initialize the retrieval usecase."""
        self.document_store = document_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_linker = entity_linker
        self.reranker = reranker

    async def execute(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        """Execute the retrieval pipeline."""
        # 1. Vector Search on Chunks
        query_embedding = await self.embedder.embed(query)
        retrieved_chunks = await self.document_store.search_chunks(query_embedding, top_k=top_k)

        # 2. Entity Linking on Query
        all_nodes = await self.graph_store.get_all_nodes()

        # Cast due to dynamic nature of linker for now, ideally linker should be a formal port
        link_entities_func = getattr(self.entity_linker, "link_entities", None)
        linked_nodes: list[Node] = []
        if link_entities_func:
            linked_nodes = await link_entities_func(query, all_nodes)

        # We can also do vector search on entities if needed
        # (Assuming graph_store has a search_nodes method, but we'll stick to linking for now)

        # 3. Multi-hop Traversal
        seed_node_ids = [n.id for n in linked_nodes]
        traversed_nodes: list[Node] = []
        traversed_edges: list[Edge] = []
        if seed_node_ids:
            traversed_nodes, traversed_edges = await self.graph_store.traverse(
                seed_node_ids, depth=2
            )

        # 4. Context Combination
        contexts = []
        for chunk in retrieved_chunks:
            contexts.append(chunk.text)

        for node in traversed_nodes:
            # Add node and its neighbors as context
            desc = f"Entity: {node.name} ({node.label}). {node.description or ''}"
            contexts.append(desc)

        # Add relationships
        for edge in traversed_edges:
            desc = f"Relationship: {edge.source_id} {edge.relation} {edge.target_id}."
            contexts.append(desc)

        # Deduplicate
        contexts = list(set(contexts))

        if not contexts:
            return []

        # 5. Reranking
        rerank_func = getattr(self.reranker, "rerank", None)
        if rerank_func:
            try:
                reranked_results = await rerank_func(query, contexts, top_k=top_k)
                return cast(list[dict[str, str | float]], reranked_results)
            except Exception:
                # Fallback if reranking fails
                pass

        return [{"text": ctx, "score": 1.0} for ctx in contexts[:top_k]]

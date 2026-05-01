"""Integration tests for SurrealDB.

Validates the integration between the Infrastructure layer (SurrealDB)
and a real database engine using testcontainers.
"""

from collections.abc import AsyncGenerator

import pytest
from surrealdb import AsyncSurreal
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from app.domain.models import Chunk, Document, Edge, Node
from app.infrastructure.database.store import (
    SurrealDocumentStore,
    SurrealGraphStore,
    _extract_rows,
)

# Mark all tests in this module
pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture(scope="function")
async def surreal_db() -> AsyncGenerator[AsyncSurreal, None]:  # type: ignore
    """Provides an authenticated SurrealDB client connected to a containerized instance.

    If the Docker container fails to start, the test is skipped.

    Yields:
        An authenticated SurrealDB client.
    """
    container = None
    db = None
    try:
        # Configuration for SurrealDB memory mode
        # NOTE: DockerContainer uses the Docker SDK, which can interface with Podman.
        # We ensure Podman compliance by relying on the environment (DOCKER_HOST).
        container = (
            DockerContainer("surrealdb/surrealdb:latest")
            .with_command("start --user root --pass root memory")
            .with_exposed_ports(8000)
        )

        container.start()
        # Wait specifically for the SurrealDB startup signal
        wait_for_logs(container, "Started web server on 0.0.0.0:8000")

        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)

        url = f"ws://{host}:{port}/rpc"
        db = AsyncSurreal(url)
        await db.connect()
        await db.signin({"username": "root", "password": "root"})
        await db.use(namespace="test", database="test")
        yield db
    except Exception as e:
        pytest.skip(f"Could not start SurrealDB test container: {e}")
    finally:
        if db:
            try:
                await db.close()
            except Exception:
                pass
        if container:
            try:
                container.stop()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_surreal_document_store_integration(surreal_db: AsyncSurreal) -> None:  # type: ignore
    """Tests full integration of SurrealDocumentStore with a real SurrealDB instance.

    Given:
        A SurrealDocumentStore connected to a real SurrealDB instance.
    When:
        A document and its corresponding chunks are saved.
    Then:
        The document and chunks should be correctly persisted and retrievable.

    Args:
        surreal_db: The authenticated SurrealDB client fixture.
    """
    # Arrange
    store = SurrealDocumentStore(surreal_db)
    doc = Document(id="doc1", filename="test.md")
    chunks = [
        Chunk(id="c1", document_id="doc1", text="Hello", index=0, embedding=[0.1, 0.2]),
        Chunk(id="c2", document_id="doc1", text="World", index=1, embedding=[0.8, 0.9]),
    ]

    # Act
    await store.save_document(doc)
    await store.save_chunks(chunks)

    # Assert: Query Document via raw SurrealQL for ground truth verification
    from surrealdb import RecordID

    doc_result = await surreal_db.query(
        "SELECT * FROM document WHERE id = $id;",
        {"id": RecordID("document", "doc1")},
    )
    rows = _extract_rows(doc_result)
    assert len(rows) == 1
    assert rows[0]["filename"] == "test.md"

    # Assert: Search Chunks
    chunk_result = await surreal_db.query("SELECT * FROM chunk ORDER BY index ASC;")
    chunk_rows = _extract_rows(chunk_result)
    assert len(chunk_rows) == 2
    assert chunk_rows[0]["text"] == "Hello"
    assert chunk_rows[1]["text"] == "World"


@pytest.mark.asyncio
async def test_surreal_graph_store_integration(surreal_db: AsyncSurreal) -> None:  # type: ignore
    """Tests full integration of SurrealGraphStore with a real SurrealDB instance.

    Given:
        A SurrealGraphStore connected to a real SurrealDB instance.
    When:
        Nodes and edges representing a knowledge graph are saved.
    Then:
        The graph structure should be correctly persisted and retrievable via store methods.

    Args:
        surreal_db: The authenticated SurrealDB client fixture.
    """
    # Arrange
    store = SurrealGraphStore(surreal_db)
    nodes = [
        Node(id="n1", label="PERSON", name="Alice"),
        Node(id="n2", label="COMPANY", name="Acme Corp"),
    ]
    edges = [Edge(source_id="n1", target_id="n2", relation="WORKS_AT", description="since 2020")]

    # Act
    await store.save_nodes(nodes)
    await store.save_edges(edges)

    # Assert: Retrieve Nodes using the store's own abstraction
    retrieved_nodes = await store.get_all_nodes()
    assert len(retrieved_nodes) == 2
    node_names = [n.name for n in retrieved_nodes]
    assert "Alice" in node_names
    assert "Acme Corp" in node_names

    # Assert: Retrieve Edges
    retrieved_edges = await store.get_all_edges()
    assert len(retrieved_edges) == 1
    assert retrieved_edges[0].source_id == "n1"
    assert retrieved_edges[0].target_id == "n2"
    assert retrieved_edges[0].relation == "WORKS_AT"

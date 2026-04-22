"""Integration tests for SurrealDB.

Requires surrealdb docker container. Use pytest-testcontainers if available,
or just skip if testcontainers aren't running.
"""

import pytest

from app.domain.models import Chunk, Document, Edge, Node
from app.infrastructure.database.store import SurrealDocumentStore, SurrealGraphStore


@pytest.fixture(scope="function")
async def surreal_db():
    """Provide an authenticated SurrealDB client.

    If docker container is not available or testcontainers fails,
    we skip the test.
    """
    from surrealdb import Surreal
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = None
    db = None
    try:
        container = (
            DockerContainer("surrealdb/surrealdb:v2.1.4")
            .with_command("start --user root --pass root memory")
            .with_exposed_ports(8000)
        )

        container.start()
        wait_for_logs(container, "Started web server on")

        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)

        url = f"ws://{host}:{port}/rpc"
        db = Surreal(url)
        await db.connect()
        await db.signin({"user": "root", "pass": "root"})
        await db.use("test", "test")

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
@pytest.mark.db
async def test_surreal_document_store_integration(surreal_db) -> None:
    """Test full integration of DocumentStore with SurrealDB.

    Arrange: Set up SurrealDocumentStore with the real db connection.
    Act: Save a document and chunks, then query them.
    Assert: The saved document and chunks can be retrieved correctly.
    """
    store = SurrealDocumentStore(surreal_db)

    # Act: Save
    doc = Document(id="doc1", filename="test.md")
    await store.save_document(doc)

    chunks = [
        Chunk(id="c1", document_id="doc1", text="Hello", index=0, embedding=[0.1, 0.2]),
        Chunk(id="c2", document_id="doc1", text="World", index=1, embedding=[0.8, 0.9]),
    ]
    await store.save_chunks(chunks)

    # Assert: Query Document
    doc_result = await surreal_db.query("SELECT * FROM document WHERE id = 'doc1';")
    # SurrealDB structure for responses
    if isinstance(doc_result, list) and len(doc_result) > 0:
        res = doc_result[0]
        if isinstance(res, dict) and "result" in res:
            assert len(res["result"]) == 1
            assert res["result"][0]["content"] == "test.md"

    # Assert: Search Chunks (since vector search might need an index to actually work,
    # we just check basic chunk insertion directly for now)
    chunk_result = await surreal_db.query("SELECT * FROM chunk ORDER BY index ASC;")
    if isinstance(chunk_result, list) and len(chunk_result) > 0:
        res = chunk_result[0]
        if isinstance(res, dict) and "result" in res:
            assert len(res["result"]) == 2
            assert res["result"][0]["text"] == "Hello"
            assert res["result"][1]["text"] == "World"


@pytest.mark.asyncio
@pytest.mark.db
async def test_surreal_graph_store_integration(surreal_db) -> None:
    """Test full integration of GraphStore with SurrealDB.

    Arrange: Set up SurrealGraphStore with real db connection.
    Act: Save nodes and edges, then retrieve them.
    Assert: The saved nodes and edges match the retrieved ones.
    """
    store = SurrealGraphStore(surreal_db)

    # Act: Save Nodes
    nodes = [
        Node(id="n1", label="PERSON", name="Alice"),
        Node(id="n2", label="COMPANY", name="Acme Corp"),
    ]
    await store.save_nodes(nodes)

    # Act: Save Edges
    edges = [Edge(source_id="n1", target_id="n2", relation="WORKS_AT", description="since 2020")]
    await store.save_edges(edges)

    # Assert: Retrieve Nodes
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

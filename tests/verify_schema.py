import asyncio
import logging

from surrealdb import AsyncSurreal
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from app.db.schema import get_schema_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_schema():
    """Verify the SurrealDB schema against a live container."""
    db = None
    container = (
        DockerContainer("surrealdb/surrealdb:v3.0.5")
        .with_command("start --user root --pass root memory")
        .with_exposed_ports(8000)
    )

    try:
        logger.info("Starting SurrealDB container...")
        container.start()
        wait_for_logs(container, "Started web server on 0.0.0.0:8000")

        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)
        url = f"ws://{host}:{port}/rpc"

        db = AsyncSurreal(url)
        await db.connect()
        await db.signin({"username": "root", "password": "root"})
        await db.use(namespace="test", database="test")

        logger.info("Connection successful. Executing schema queries...")
        queries = get_schema_queries()
        for i, query in enumerate(queries):
            logger.info(f"Executing query {i + 1}/{len(queries)}")
            await db.query(query)

        logger.info("Schema initialization successful! New syntax is accepted.")
    except Exception:
        logger.exception("Schema verification FAILED")
        raise
    finally:
        if db is not None:
            await db.close()
        container.stop()


if __name__ == "__main__":
    asyncio.run(verify_schema())

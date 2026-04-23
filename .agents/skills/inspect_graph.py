# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "surrealdb",
# ]
# ///
# Skill Description: Runs a read-only SurrealQL query against the local database to verify entity nodes or edges.

import asyncio
import sys
from surrealdb import Surreal


async def main():
    if len(sys.argv) < 2:
        print("Error: Provide a SurrealQL query (e.g., 'SELECT * FROM entity LIMIT 5;').")
        sys.exit(1)

    query = sys.argv[1]

    # Connecting to the local Podman instance defined in your rules
    async with Surreal("ws://127.0.0.1:8000/rpc") as db:
        await db.signin({"user": "root", "pass": "root"})
        await db.use("omni", "copilot")

        try:
            result = await db.query(query)
            print(f"Query Result:\n{result}")
        except Exception as e:
            print(f"Query Failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

# Agent Context & State

## Current Focus

- Implementing the missing HNSW vector indices in `app/infrastructure/database/schema.py`.
- Replacing dummy embedders with `SentenceTransformerEmbedder` in `app/infrastructure/`.

## Known System Quirks

- The `fastcoref` library currently throws an exception on `get_clusters(as_strings=False)`.
- Port 8080 occasionally hangs during local restarts. Always verify it is free (`fuser -k 8080/tcp`) before starting Uvicorn.

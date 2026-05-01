---
description: Builds the Semantic Blocking + Cross-Encoder merge pipeline.
---

# Implement Entity Resolution Pipeline

Usage: /implement-resolution

## Step 1: Infrastructure - The Arbitrator

Update `app/infrastructure/embeddings.py`:

- Integrate `CrossEncoder` from the `sentence-transformers` library (using `BAAI/bge-reranker-v2-m3`).
- Implement a `verify_similarity(text_a, text_b)` method that returns a float score.
- Ensure the model is cached locally to avoid redundant downloads.

## Step 2: Infrastructure - The Graph Merge Port

Update `app/infrastructure/database/graph_store.py`:

- Add an async `merge_nodes(canonical_data, source_ids)` method.
- **SurrealQL Requirement:** Use a transaction.
- Use `array::distinct(array::concat(...))` to merge `source_chunk_ids`.
- Implement the "Edge Rewiring" logic to move relationships from `source_ids` to the new `canonical_id`.

## Step 3: Application Layer - The Orchestrator

Create `app/application/resolution.py`:

- Define `EntityResolutionUseCase`.
- **Logic:**
  1. Query SurrealDB for entities with high cosine similarity (Semantic Blocking).
  2. For each pair, run the `verify_similarity` Cross-Encoder.
  3. If score > 0.95, call the `merge_nodes` repository method.

## Step 4: Verification

- Create a test in `tests/integration/test_resolution.py`.
- Mock two entities ("USA", "United States") with different `source_chunk_ids`.
- Assert that after resolution, only one node exists and its `source_chunk_ids` contains the union of the originals.

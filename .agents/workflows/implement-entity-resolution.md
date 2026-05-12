---
description: Builds the Semantic Blocking + Cross-Encoder merge pipeline.
---

# Implement Entity Resolution Pipeline

Usage: /implement-resolution

## Step 1: Slice - Domain Models & Arbitrator

Update `app/pipelines/resolution/models.py`:
- Define `ResolutionState` and `EntityMatch` Pydantic models.

Update `app/pipelines/resolution/arbitrator.py`:
- Integrate `CrossEncoder` from the `sentence-transformers` library (using `BAAI/bge-reranker-v2-m3`).
- Implement a `verify_similarity(text_a, text_b)` method that returns a float score.
- Ensure the model is cached locally.

## Step 2: Slice - Graph Persistence

Update `app/pipelines/resolution/persistence.py`:
- Implement the `merge_nodes(canonical_data, source_ids)` logic.
- **SurrealDB 3.x Pattern**: Use the `SurrealGraphStore` implementation established in Phase 6.
- Use `RELATE` and `UPSERT` with `array::distinct(array::concat(...))` to merge `source_chunk_ids`.
- Implement "Edge Rewiring" to redirect relationships from `source_ids` to the `canonical_id`.

## Step 3: Slice - Orchestrator

Update `app/pipelines/resolution/orchestrator.py`:
- Define `EntityResolutionUseCase`.
- **Logic:**
  1. Query SurrealDB for entities with high cosine similarity (Semantic Blocking).
  2. For each pair, run the `verify_similarity` Cross-Encoder.
  3. If score > 0.95, call the `merge_nodes` persistence method.

## Step 4: Verification

- Create a test in `tests/integration/test_resolution.py`.
- Mock two entities ("USA", "United States") with different `source_chunk_ids`.
- Assert that after resolution, only one node exists and its `source_chunk_ids` contains the union of the originals.

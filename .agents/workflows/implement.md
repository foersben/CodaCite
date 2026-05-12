---
description: Safely implement a new feature from planning to testing.
---

# Vertical Slice Feature Implementation

Usage: /implement [description of feature]

## Step 1: Planning (Artifact)

Analyze the request against our **Vertical Slice Architecture** rules. Generate an "Implementation Plan" Artifact detailing:

1. New slice directory: `app/pipelines/<feature_name>/`
2. Domain Models & Logic: Residing within the slice.
3. Core Extensions: If any shared interfaces in `app/core/` need updating.
4. API Integration: New routes in `app/api/` or updates to existing ones.

**MANDATORY**: If the feature involves ingestion, specify the structural chunking strategy and character offset preservation logic.

Stop and wait for the user to approve the plan.

## Step 2: Execution

Once approved, write the code strictly adhering to the plan. Run `uv run ruff check --fix app` and `uv run mypy app` to ensure formatting and typing are correct.

## Step 3: Verification

Trigger the `/qa-pass` workflow on the newly created or modified pipeline files to ensure the feature works.

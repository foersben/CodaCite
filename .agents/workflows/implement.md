---
description: Safely implement a new feature from planning to testing.
---

# Hexagonal Feature Implementation

Usage: /implement [description of feature]

## Step 1: Planning (Artifact)

Analyze the request against our Hexagonal Architecture rules. Generate an "Implementation Plan" Artifact detailing:

1. Pydantic Models to add to `app/domain/`
2. Interface updates in `app/infrastructure/`
3. Use case logic in `app/application/`
4. FastAPI routes in `app/interfaces/`

Stop and wait for the user to approve the plan.

## Step 2: Execution

Once approved, write the code strictly adhering to the plan. Run `uv run ruff check --fix app` to ensure formatting is correct.

## Step 3: Verification

Trigger the `/qa-pass` workflow on the newly created or modified application logic files to ensure the feature works.

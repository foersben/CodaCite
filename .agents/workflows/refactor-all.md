---
description: Safely executes a codebase-wide refactor by combining global automated tooling with iterative, folder-by-folder AI auditing to avoid context window exhaustion.
---

# Repository-Wide Refactor Orchestration

Usage: /refactor-all

## Step 1: Global Automated Pass (Terminal)

Before doing any manual AI auditing, use your terminal tools to fix the low-hanging fruit across the entire codebase at once:

1. Run `uv run ruff check --fix app tests`
2. Run `uv run ruff format app tests`
3. Run `uv run mypy app`

If these commands fail, fix the global errors before proceeding to Step 2.

## Step 2: Strategic Chunking

Do not attempt to read or rewrite all files at once. You will tackle the codebase strictly in this Hexagonal order:

1. `app/domain/` (Core logic, models, ports)
2. `app/infrastructure/` (Database, Embedders, Gemini)
3. `app/application/` (Use Cases)
4. `app/interfaces/` (FastAPI Routers)

## Step 3: Iterative Execution

For each folder listed in Step 2:

1. Use your filesystem tools to list the Python files in that specific folder.
2. Apply the strict Hexagonal Architecture constraints and Google-style docstring rules (as defined in our standard refactor rules) to those specific files.
3. **CRITICAL:** Stop and output a brief "Folder Status" Artifact summarizing what you changed in that specific folder.
4. Ask the user for permission to proceed to the next folder. Do not proceed until the user says "continue".

## Step 4: Final Verification

Once all folders are complete, run `uv run pytest` to ensure the massive refactoring effort did not break the GraphRAG logic or API routes.

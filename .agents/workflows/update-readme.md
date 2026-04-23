---
description: Synchronizes the root README.md with the latest code, architectural changes, and agent workflows.
---

# Update README Documentation

Usage: /update-readme

## Step 1: Context Gathering

Analyze the `docs/architecture.md`, `.agents/rules/`, and the core `app/main.py` file to identify any new capabilities, endpoints, or infrastructure shifts.

## Step 2: Content Alignment

Review the current `README.md`. You MUST ensure that:

- The "Getting Started" or "Installation" sections correctly mandate `uv` and `podman` (and strictly forbid `pip` or `docker`).
- The "Architecture" section accurately reflects the Hexagonal design and SurrealDB GraphRAG implementation.
- The "Agent Workspace" section lists the available workflows (like `/qa-pass`, `/implement`).

## Step 3: Artifact Generation

Do not immediately overwrite the file. Generate a "README Diff" Artifact showing what sections you intend to add, modify, or remove. Wait for user approval before writing to the filesystem.

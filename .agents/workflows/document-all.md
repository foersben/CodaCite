---
description: A batch orchestration workflow that updates inline docstrings across all files and synchronizes the global project documentation.
---

# Repository-Wide Documentation Sync

Usage: /document-all

## Step 1: Inline Docstring Sweep (Iterative)

You will process the codebase folder-by-folder (`domain`, `infrastructure`, `application`, `interfaces`):

1. Read the Python files in the current folder.
2. Ensure every module, class, and function has a strict Google-style docstring.
3. If the docstring is missing or outdated, rewrite it to accurately reflect the current Hexagonal business logic.
4. **CRITICAL:** Pause after completing a folder. Output a brief Artifact summarizing which files were documented. Wait for user approval to proceed.

## Step 2: Global Architecture Sync

Once the code is fully documented, read `docs/architecture.md` and `docs/infrastructure.md`.
Update these markdown files if you discovered any new patterns, endpoints, or database schema additions during Step 1 that are not currently reflected in the docs.

## Step 3: Zensical & README Alignment

Trigger the underlying logic of `/sync-zensical` and `/update-readme`.
Ensure that `zensical.toml` tracks all the newly documented modules and that the root `README.md` is perfectly up to date.

## Step 4: Verification

Run `uv run ruff check --fix app` to ensure your docstring additions did not introduce any formatting or line-length linting errors.

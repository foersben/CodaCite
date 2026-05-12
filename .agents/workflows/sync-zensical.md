---
description: Ensures the zensical.toml file accurately reflects the current state of the application and documentation.
---

# Sync Zensical Configuration

Usage: /sync-zensical

## Step 1: Slice & Asset Discovery

1. **Scan Pipelines**: List all subdirectories in `app/pipelines/`.
2. **Scan Docs**: List all `.md` files in `docs/`.
3. **Read Metadata**: Read `zensical.toml` and `pyproject.toml`.

## Step 2: Intelligent Update

1. **Version Sync**: Ensure `project.version` in `zensical.toml` matches `project.version` in `pyproject.toml`.
2. **Slice Audit**:
   - Ensure every subdirectory in `app/pipelines/` has a corresponding `[tracking.<slice_name>]` table.
   - For each slice, verify that the `files` array includes all primary Python files in that directory.
3. **Dead Link Cleanup**: Remove any files from `zensical.toml` that no longer exist in the workspace.
4. **Core Tracking**: Ensure `app/core/` and `app/db/` assets are tracked under `[tracking.core]`.

## Step 3: Navigation & TOML Validation

1. **Nav Sync**:
   - Cross-reference `nav` list in `zensical.toml` with the files in `docs/`.
   - **WARNING**: If a documentation file exists in `docs/` but is not in the `nav` list, alert the user (do not auto-add unless it is a standard index file).
2. **Format Check**: Use `uv run python -c "import tomllib; tomllib.loads(open('zensical.toml').read())"` to verify validity.

## Step 4: Reporting

Output a Markdown Table Artifact summarizing:
- Added/Removed files.
- Version updates.
- Navigation discrepancies.

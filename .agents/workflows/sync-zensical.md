---
description: Ensures the zensical.toml file accurately reflects the current state of the application and documentation.
---

# Sync Zensical Configuration

Usage: /sync-zensical

## Step 1: Analysis

Read the `zensical.toml` file. Cross-reference its contents with the current project metadata in `pyproject.toml` and the structure of the `docs/` directory.

## Step 2: Update Configuration

Identify any discrepancies. If new Python modules have been added to `app/` or new markdown files have been added to `docs/` that should be tracked, update the `zensical.toml` file to include them. 

## Step 3: Formatting & Validation

Ensure the updated file maintains strictly valid TOML formatting. 

## Step 4: Reporting

Output a brief summary Artifact of which tables or keys in `zensical.toml` were updated and why.

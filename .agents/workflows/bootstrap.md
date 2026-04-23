---
description: Cleans and rebuilds the environment in writable project space.
---

# Project Bootstrap & Environment Fix

Usage: /bootstrap


## Step 1: Create Writable Sandbox

1. Run: `mkdir -p .uv_cache .uv_python`
2. Export the following to bypass read-only system paths:
  - `export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv_python`
  - `export UV_CACHE_DIR=$(pwd)/.uv_cache`

## Step 2: Clean Slate

Remove any existing, broken virtual environments:
`rm -rf .venv`

## Step 3: Reconstruction

1. Create a fresh virtual environment: `uv venv --python 3.11`
2. Synchronize all dependencies: `uv sync --all-extras`
3. Verify installations: `uv run python -c "import fastapi; import pydantic_settings; print('Environment Stable')"`

## Step 4: Completion

Output a confirmation Artifact.

---
description: Runs the standard Antigravity pre-commit and test checks.
---

# Test runner

## Step 1
Run the ruff linter and formatter using `uv run ruff check --fix app tests` and `uv run ruff format app tests`. Ensure all issues are fixed.

## Step 2
Run the type checker using `uv run mypy app`. If there are any `Any` types outside of 3rd party boundaries, refactor them.

## Step 3
Run the test suite using `uv run pytest`. If any tests fail, identify the failure, fix the underlying code, and re-run the tests until they pass.

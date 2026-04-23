---
description: Generates and verifies unit tests for a specific target file.
---

# Autonomous QA Pass

Usage: /qa-pass [target_file.py]

## Step 1: Analysis

Analyze the provided `target_file.py`. Identify the core business logic, edge cases, and any infrastructure dependencies that need to be mocked.

## Step 2: Test Generation

Create a new test file in the appropriate `tests/unit/` or `tests/integration/` directory. 

- You MUST follow the Arrange, Act, Assert (AAA) pattern.
- You MUST use `pytest-mock` to mock any external database or API calls.
- You MUST use `@pytest.mark.asyncio` for async functions.

## Step 3: Execution & Iteration

Use your terminal tool to run: `uv run pytest [path_to_new_test_file]`.

- If the tests fail, analyze the output, fix the tests (or the source code if a bug was found), and re-run.
- Repeat this step autonomously up to 3 times until the tests pass.

## Step 4: Reporting

Generate a Test Summary Artifact detailing coverage and what edge cases were handled.

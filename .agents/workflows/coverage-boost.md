---
description: Audits the repository for missing test coverage and autonomously writes pytest tests to reach a specific target (default 90%+).
---

# Autonomous Coverage Boost

Usage: /coverage-boost [target_percentage]

## Step 1: The Gap Analysis

Use the terminal tool to run: `uv run pytest --cov=app --cov-report=term-missing`.

1. Identify the files with the lowest coverage.
2. Read the source code of those files.
3. Identify "Dark Logic": Branch conditions (if/else), exception handlers, and edge cases that are not currently hit by tests.

## Step 2: Test Strategy (Artifact)

Generate a "Coverage Strategy" Artifact. Group the needed tests by file and type:

- **Unit Tests:** For pure logic in `app/domain` or `app/application`.
- **Integration Tests:** For `app/infrastructure` (SurrealDB/Gemini) using proper mocks.

**Strict Rule:** Use `pytest` only. NO `unittest`.

## Step 3: Iterative Generation

For each file identified in the strategy:

1. Generate the missing test cases in the corresponding `tests/` directory.
2. Follow the AAA (Arrange, Act, Assert) pattern.
3. Ensure Google-style docstrings are present.
4. Run `uv run pytest [new_test_file]` to verify.

## Step 4: Final Validation

Run the global coverage command again.

- If the `target_percentage` is reached: Output the final "Coverage Success" Artifact.
- If not: Repeat from Step 1 focusing on the remaining gaps.

---
description: Autonomously audits the codebase for missing test coverage and iteratively generates unit tests layer-by-layer.
---

# Repository-Wide QA Sweep

Usage: /qa-pass-all

## Step 1: Global Coverage Audit (Terminal)

Use your terminal tools to run `uv run pytest --cov=app --cov-report=term-missing`. 
Analyze the terminal output to identify exactly which files in `app/` are missing test coverage.

## Step 2: Strategic Chunking

Do not attempt to write tests for all missing files at once. You will tackle the missing coverage strictly in this Hexagonal order:

1. `app/domain/` (Models and pure logic)
2. `app/infrastructure/` (Database and external API mocks)
3. `app/application/` (Use cases)
4. `app/interfaces/` (FastAPI routes)

## Step 3: Iterative Test Generation

For the current folder in the sequence:

1. Select the files identified in Step 1 that lack coverage.
2. Apply the core `/qa-pass` logic to each file: Create tests in `tests/` following the AAA pattern, using `pytest-mock`, and `@pytest.mark.asyncio`.
3. Run `uv run pytest` to ensure the newly generated tests pass.
4. **CRITICAL:** Stop and output a "Coverage Update" Artifact summarizing the new tests written for this specific folder. Wait for the user to say "continue" before moving to the next folder.

## Step 4: Final Reporting

Once all folders are complete, run the coverage command again and output a final "QA Pass All Summary" Artifact showing the improved percentage.

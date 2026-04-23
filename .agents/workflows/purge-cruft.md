---
description: Iteratively identifies and removes outdated code, redundant "AI-generated" wrappers, and legacy bridges while maintaining functional integrity.
---

# AI Technical Debt Purge

Usage: /purge-cruft

## Step 1: Cruft Audit

Scan the codebase for the following technical debt patterns:

- **Redundant Wrappers:** Functions that simply call another function without adding logic.
- **Dead Bridges:** "Backward compatibility" code that is no longer imported or used by any active module.
- **AI Verbosity:** Overly defensive code or comments that explain obvious logic instead of domain intent.
- **Duplicated Logic:** Snippets that were "copy-pasted" by the AI across different layers.

## Step 2: Incremental Deletion

Do not delete everything at once. Select one module or architectural layer (e.g., `infrastructure`).

1. Identify a candidate for removal or simplification.
2. Perform the deletion/refactor.
3. Use the terminal to run `uv run ruff check --fix` to ensure no broken imports were left behind.

## Step 3: Test-Driven Validation

Immediately run `uv run pytest` for the affected module.

- **If Tests Pass:** Proceed to the next candidate in Step 2.
- **If Tests Fail:** **Evaluate the Failure.**
    - **Scenario A (Accidental Loss):** You removed logic that was actually essential. ACTION: Revert the change.
    - **Scenario B (Tangled Dependency):** The code was "cruft" but its removal exposed a tightly coupled dependency that needs proper untangling. ACTION: Do NOT revert. Refactor the dependent code to work without the cruft.
    - **Scenario C (Outdated Test):** The test itself was relying on the legacy behavior. ACTION: Update the test.

## Step 4: Artifact Summary

After each layer is cleaned, output a "Purge Report" Artifact:

- **Removed:** List of functions/classes deleted.
- **Simplified:** Logic that was compressed.
- **Structural Fixes:** Dependencies that were untangled.

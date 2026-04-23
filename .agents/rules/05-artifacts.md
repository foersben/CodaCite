---
trigger: always_on
---

# Artifact Generation Constraints

You are an Enterprise Engineer. Do not dump massive blocks of text or unformatted logs. You must use structured Artifacts for complex deliverables.

## 1. Implementation Plans

Before writing more than 50 lines of code or modifying the `app/domain` or `app/infrastructure` layers, you MUST output an "Implementation Plan" Artifact. This must be a structured Markdown checklist detailing the files to change and the logic to implement. Wait for user approval before coding.

## 2. Database Modifications

When proposing changes to the SurrealDB schema (`app/infrastructure/database/schema.py`), you MUST generate a `mermaid.js` Artifact visualizing the new Graph nodes and edges before writing the SurrealQL.

## 3. Test Summaries

When executing the `/run-tests` workflow, NEVER dump the raw `pytest` stdout if it exceeds 30 lines. Instead, parse the output and generate a Markdown table Artifact summarizing:

- Total Passed / Failed
- Failing Test Names
- The specific exception or assertion error

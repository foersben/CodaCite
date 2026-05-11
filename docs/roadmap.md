# CodaCite Engineering Roadmap

This roadmap outlines the strategic evolution of the CodaCite engine, transitioning from a stable, unit-tested foundation toward a high-performance, multi-modal knowledge platform.

---

## Phase 1: Stabilization & Manual Verification (UAT)

Before proceeding with new feature development, the core pipeline must be validated through "Real-World" stress testing.

### Step 1: The "Real-World" Smoke Test

1. **Ingestion Stress Test**: Upload a medium-sized document (10+ pages). Verify the `StructuralContextChunker` performance and verify that VRAM-aware routing correctly selects the GPU/CPU/MPS device.
2. **Factual Precision Check**: Ask hyper-specific questions. Verify that the response includes verbatim quotes and correct `[n]` citations.
3. **Negative Constraint Check**: Ask questions about non-existent information. Verify the fallback phrase: *"The provided documents do not contain information about..."*
4. **UI Citation Rendering**: Ensure the frontend properly handles Server-Sent Events (SSE) and renders citation badges correctly.

---

## Phase 2: Performance & Infrastructure Optimization

Focusing on the next set of computational bottlenecks identified during the stabilization phase.

### Mission A: Graph/Entity Extraction Optimization

* **Objective**: Reduce the CPU overhead of zero-shot NER (GLiNER) and relationship mapping.
* **Key Tasks**: Implement efficient batching in `app/pipelines/extraction/`, drop low-value chunks before processing, and optimize context window usage for relationship mapping.

### Mission C: Global Summarization Tuning

* **Objective**: Optimize Map-Reduce logic for local 7B models.
* **Key Tasks**: Refactor `app/pipelines/ingestion/summarization.py` to use rolling summaries or high-density chunk batching to maintain low temperatures and fast generation times.

### SurrealDB 3.x Migration

* **Objective**: Fully transition to O(1) RecordID fetching across all persistence nodes.
* **Key Tasks**: Audit existing `surrealql` queries in `app/db/` and `app/pipelines/` to eliminate join latency.

---

## Phase 3: User Experience & Visualization

Bridging the gap between backend intelligence and user-facing clarity.

### Mission B: UI/UX Citation Guardrail Visualization

* **Objective**: Visually represent backend hallucination detection.
* **Key Tasks**: Update `index.html` and `style.css` to handle `{"verified": false, "warning": true}` payloads. Implement red citation badges or warning icons (⚠️) for unverified quotes.

---

## Phase 4: Advanced AI Capabilities

Expanding the system's cognitive breadth beyond standard text-based RAG.

### Phase 7: Multi-Modal Contextualization

* **Objective**: Integrate Vision-Language Model (VLM) capabilities into the RAG pipeline.
* **Key Tasks**: Enable processing of charts, diagrams, and images within PDFs using models like `Qwen2-VL` or `Pixtral`.

### Entity Resolution Optimization

* **Objective**: Implement advanced Semantic Blocking and Cross-Encoder merge logic for "Modular Monolith" entity resolution.
* **Key Tasks**: Finalize the extraction and integration of the resolution logic into the main ingestion flow.

---

## Execution Blueprints (SOPs)

The following blueprints represent verified implementation strategies for key infrastructure upgrades.

### Blueprint: Dynamic VRAM Allocation for Docling

* **Safe Torch Imports**: Wrap `import torch` in `try/except`.
* **Optimal Device Selection**: Helper method to probe `torch.cuda` (>1.5GB) or `torch.backends.mps`.
* **Fallback**: Default to `"cpu"` with appropriate logging.

### Blueprint: Local Generator Prompting & Depth

* **System Prompt Hardening**: Rule 4: *"Do NOT output prefixes like 'Answer:' or 'Response:'."*
* **Context Expansion**: Set `top_k` to 6 for local models to ensure relationship intersection (e.g., Benjamin AND Peter).
* **Guardrail Resilience**: Ensure `verified: False` flags do not suppress text output in the UI.

---

## Phase 5: Agentic Infrastructure & Engineering Excellence

### Mission D: Persona & Rule Realignment

* **Objective**: Synchronize the Antigravity agent configuration with the Vertical Slice Architecture.
* **Key Tasks**:
    * Update `.agents/rules/00-global-invariants.md` to strictly enforce feature slices and prohibit deprecated hexagonal directories.
    * Realign `.agents/rules/02-backend-architect.md` to target `app/pipelines/` exclusively.
    * Implement `.agents/rules/08-ingestion-engineer.md` to protect structural chunking integrity and character offset mapping.

### Mission E: SurrealDB 3.x Pattern Standardization

* **Objective**: Optimize the developer experience for graph-based queries.
* **Key Tasks**:
    * Update `.agents/rules/01-surreal-dba.md` with the **O(1) Direct ID Retrieval** pattern.
    * Enforce strict typing for `RecordID` objects in all pipeline queries.

### Mission F: Autonomous Synchronization Workflow

* **Objective**: Automate repository-wide consistency audits.
* **Key Tasks**:
    * Enhance `/sync-zensical` to perform autonomous slice discovery and navigation audits.
    * Integrate `uv run ruff` and `uv run mypy` checks into the `/commit` workflow to ensure zero-bypass policy enforcement.

# 🤖 Antigravity: The Agentic Core

[![Agent Status](https://img.shields.io/badge/Agent-Autonomous-brightgreen.svg)](#)
[![Workflow Count](https://img.shields.io/badge/Workflows-15-blue.svg)](#workflow-directory)
[![Rules](https://img.shields.io/badge/Governance-Strict-red.svg)](#agent-rules--governance)

Welcome to the heart of the **CodaCite** autonomous development environment. This project isn't just "AI-assisted"—it is **AI-managed**. The **Antigravity** persona is an autonomous engineer that lives within this repository, enforcing strict architectural standards and quality gates.

---

## 🎯 Our Mission

To build the world's most reliable GraphRAG engine using a perfectly isolated, verifiable, and self-documenting codebase. Antigravity ensures that every line of code respects the **Hexagonal Architecture** and passes exhaustive QA before it even reaches a human reviewer.

---

## 🛠️ Workflow Directory

Antigravity operates via specialized slash commands. These aren't just scripts; they are multi-phase cognitive loops.

### 🚀 Development & Feature Implementation
| Command | Description | When to use? |
| :--- | :--- | :--- |
| `/implement` | End-to-end feature creation. | When you need a new feature from scratch. |
| `/refactor-all` | Codebase-wide architectural shifts. | When changing core patterns or dependencies. |
| `/implement-entity-resolution` | Specialized merge logic building. | For refining how the KG deduplicates data. |

### 🧪 Quality Assurance & Testing
| Command | Description | When to use? |
| :--- | :--- | :--- |
| `/run_tests` | The standard quality gate. | Before every commit or after any change. |
| `/qa-pass` | Targeted unit test generation. | When a specific file needs 100% coverage. |
| `/coverage-boost` | Global test coverage auditing. | When the total project coverage dips. |

### 🧹 Maintenance & Hygiene
| Command | Description | When to use? |
| :--- | :--- | :--- |
| `/purge-cruft` | AI boilerplate removal. | To keep the codebase lean and "human-grade". |
| `/document-all` | Global docstring synchronization. | Before major releases or after refactors. |
| `/reflect` | Post-mortem learning. | After a bug is fixed to prevent regression. |

---

## 📜 Agent Rules & Governance

The agent's "brain" is guided by the markdown files in the [rules](rules/) directory.

> [!CAUTION]
> **Zero-Bypass Policy**: Antigravity is strictly forbidden from using `--no-verify`. Code cannot be committed unless `ruff` and `mypy` pass with zero warnings.

### Core Architectural Invariants:
1.  **Domain Purity**: `app/domain` must have **zero** external dependencies (no FastAPI, no SurrealDB).
2.  **Mock-Only Testing**: Infrastructure tests must never hit live APIs; they must use the `pytest-mock` isolation layer.
3.  **Conventional Commits**: All history must follow the [Conventional Commits](https://www.conventionalcommits.org/) spec.

---

## 💡 Usage Examples

To trigger Antigravity, simply mention a workflow in your request:

- **"Add a new endpoint for notebook exports @[/implement]"**
- **"I think the database logic is getting messy, can you clean it up? @[/purge-cruft]"**
- **"Why did the last ingestion fail? Let's fix it and learn. @[/reflect]"**

---

👉 **Back to [Product README](../README.md)**

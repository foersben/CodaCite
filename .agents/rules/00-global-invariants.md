---
trigger: always_on
---

# Antigravity Global Invariants

This rule supersedes all other instructions.

## 1. Tooling Constraints

- **Package Manager:** Use `uv`.
- **Environment:** You MUST set these environment variables in your session before running any `uv` commands:
  `export UV_CACHE_DIR=$(pwd)/.cache/uv`
  `export UV_PYTHON_INSTALL_DIR=$(pwd)/.cache/uv_python`
- **Virtual Env:** Always use the local `.venv` directory. If broken, run `rm -rf .venv && uv venv --python 3.13`.
- **Container Engine:** Use `podman` and `podman-compose` ONLY.
- **Database Versioning**: We are strictly using **SurrealDB v3.0.5** with the modern Rust-based Python SDK. Do NOT use legacy 1.x syntax or assume 1.x API response envelopes.

## 2. Markdown & Documentation Standards

- **List Formatting**: Every item list or enumeration MUST be preceded and followed by a blank line. Failure to do so causes rendering issues in the documentation. Do not add blank lines between items in the same list.
- **Heading Hierarchy**: Maintain a clean, logical heading structure (H1 -> H2 -> H3).

## 3. Code Quality & Typing

- Code must pass `uv run ruff check app tests` and `uv run mypy app`.
- Use modern Python 3.13+ syntax (`str | None` instead of `Optional[str]`, `list` instead of `List`).
- `Any` is strictly prohibited unless interfacing with untyped 3rd-party libraries.

## 4. Architecture: Vertical Slice

Adhere strictly to **Vertical Slice Architecture** (Modular Monolith).

- **Feature Slices**: All business logic, models, and domain rules for a specific feature reside in `app/pipelines/<feature_name>/`.
- **Core Layer**: Shared utilities, foundational interfaces, and common logic reside in `app/core/`.
- **Prohibited Directories**: Do NOT recreate the following legacy "Hexagonal" directories:
  - `app/domain/`
  - `app/infrastructure/`
  - `app/application/`
  - `app/interfaces/`
  - `app/ingestion/` (Use `app/pipelines/ingestion/` instead)

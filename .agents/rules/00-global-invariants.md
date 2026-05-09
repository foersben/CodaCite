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
- **Database Versioning:** We are strictly using **SurrealDB v3.0.5** with the modern Rust-based Python SDK. Do NOT use legacy 1.x syntax or assume 1.x API response envelopes.

## 2. Code Quality & Typing

- Code must pass `uv run ruff check app tests` and `uv run mypy app`.
- Use modern Python 3.13+ syntax (`str | None` instead of `Optional[str]`, `list` instead of `List`).
- `Any` is strictly prohibited unless interfacing with untyped 3rd-party libraries.

## 3. Architecture

- Adhere strictly to **Hexagonal Architecture**.
- `app/domain/`: Pure logic and Pydantic models. NO external dependencies.
- `app/infrastructure/`: Concrete implementations.
- `app/interfaces/`: FastAPI routers.
- `app/application/`: Use cases coordinating the above.

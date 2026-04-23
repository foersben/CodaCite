# Agent Coding Standards

## 1. Architecture

- Adhere strictly to **Hexagonal Architecture**.
  - `app/domain/`: Pure logic and Pydantic models. NO external dependencies.
  - `app/infrastructure/`: Concrete implementations (e.g., SurrealDB, Gemini API).
  - `app/interfaces/`: FastAPI routers, dependencies, and middlewares.
  - `app/application/`: Use cases coordinating the above.

## 2. Python & Typing

- Code must pass `uv run ruff check app tests` and `uv run mypy app`.
- Use modern Python 3.11+ syntax (`str | None` instead of `Optional[str]`, `list` instead of `List`).
- `Any` is strictly prohibited unless interfacing with untyped 3rd-party libraries.

## 3. Testing

- Follow the **Arrange, Act, Assert (AAA)** pattern.
- Use `@pytest.mark.asyncio` for all I/O bound tests.
- Never hit live APIs in unit tests; use `pytest-mock` to isolate infrastructure.

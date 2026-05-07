# Agent Coding Standards

## 1. Architectural Guardrails: Vertical Slice

CodaCite follows a **Vertical Slice Architecture** (Modular Monolith). Agents MUST adhere to these organizational principles:

- **Feature Slices**: All business logic resides in `app/pipelines/<feature_name>/`.
- **Autonomous Modules**: Each slice should be self-contained. Shared utilities go into `app/core/`.
- **Ports & Adapters**: While encapsulated in slices, we still use Dependency Injection (DI). Define protocols in `app/core/interfaces.py` and implement them as injected dependencies.
- **Data Models**: Pure domain models reside in `app/models/models.py`.

### Prohibited Directories
The following legacy "Hexagonal" directories are DEPRECATED and must not be recreated:
- `app/application/`
- `app/domain/`
- `app/infrastructure/`
- `app/interfaces/`
- `app/ingestion/` (Use `app/pipelines/ingestion/` instead)

## 2. Python & Typing Constraints

- **Python 3.13+**: Use modern syntax (`str | None`, `list[str]`).
- **Strict Typing**: All functions MUST have complete type hints. `Any` is strictly prohibited.
- **Linting**: Code must pass `uv run ruff check app tests` and `uv run mypy app`.

## 3. Testing & Verification

- **AAA Pattern**: Follow Arrange, Act, Assert.
- **Asyncio**: Use `@pytest.mark.asyncio` for all I/O bound tests.
- **Mocking**: Never hit live APIs in unit tests; use `pytest-mock` or `respx` to isolate external services.
- **Database**: Use a temporary SurrealDB instance for integration tests, or mock the `SurrealGraphStore` for unit tests.

## 4. Documentation

- **Google Style**: All public methods MUST have Google-style docstrings.
- **Textbook Tone**: Documentation updates must maintain the pedagogical, formal tone established in the `/docs` directory.

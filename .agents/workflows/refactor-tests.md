---
description: Audits and refactors the entire test suite to ensure strict adherence to pytest standards, proper organization, and Google-style documentation.
---

# Standardized Test Suite Refactor

Usage: /refactor-tests

## Step 1: Discovery & Consolidation

Scan the entire repository for any files containing test logic (searching for `test_*.py` or classes inheriting from `unittest.TestCase`).

- **Standardization:** If any test files are found outside the `tests/` directory, move them into the appropriate subdirectory.
- **Restructure:** Make sure if packages contain a large number of modules to plan and execute proper restucturing in different packages and subpackages within `tests/`.
- **Package Integrity:** Ensure every directory within `tests/` contains an `__init__.py` file to maintain proper Python package structure.

## Step 2: Pytest Framework Conversion (Strict)

Audit every test file for legacy `unittest` patterns. You MUST:

- **Remove Class Inheritance:** Convert any `class TestX(unittest.TestCase)` into standard `pytest` classes (no inheritance) or standalone functions.
- **Assertion Rewrite:** Replace all `self.assert*` calls (e.g., `self.assertEqual`, `self.assertTrue`) with standard Python `assert` statements.
- **Fixture Migration:** Convert `setUp` and `tearDown` methods into scoped `pytest` fixtures (defined in the file or `tests/conftest.py`).

## Step 3: Pytest Feature Optimization

Enhance the tests using advanced `pytest` features:

- **Parameterization:** Use `@pytest.mark.parametrize` to replace redundant loops within tests.
- **Markers:** Apply `@pytest.mark.asyncio` to all async tests and ensure integration tests are marked with `@pytest.mark.integration`.
- **Mocks:** Standardize on `pytest-mock` (the `mocker` fixture) for all mocking needs, replacing `unittest.mock.patch`.

## Step 4: Naming & Google-Style Documentation

- **Naming:** Ensure all test files start with `test_` and all test functions start with `test_`. Names must be descriptive (e.g., `test_ingestion_fails_on_empty_file`).
- **Docstrings:** Add Google-style docstrings to every test module and complex test function.
    - The module docstring must explain what layer of the Hexagonal architecture is being tested.
    - Function docstrings should briefly explain the scenario being validated (Given/When/Then).

## Step 5: Verification & Cleanup

1. Run `uv run ruff check --fix tests/` to ensure the new test code is linted.
2. Run `uv run ruff format tests/`.
3. **Execution:** Run the entire suite using `uv run pytest`.
4. **Artifact:** Output a "Test Refactor Report" Artifact listing:
    - Files moved or renamed.
    - Count of `unittest` patterns removed.
    - Final `pytest` execution summary (Pass/Fail).

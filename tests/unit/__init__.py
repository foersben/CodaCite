"""Unit testing package.

Scope: Fast, isolated tests focusing on individual functions and classes.
Rules: No external network or database calls (no real databases, no real LLM APIs).
Everything external must be mocked (e.g., using AsyncMock). Local filesystem fixtures
like `tmp_path` are permitted for isolated file operations.
"""

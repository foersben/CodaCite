"""Integration testing package.

Scope: Tests that verify the interaction between multiple components.
Rules: May use local databases (e.g., testcontainers) or local file I/O, but
should still avoid external network calls like live LLM APIs unless explicitly marked.
"""

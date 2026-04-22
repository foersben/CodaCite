"""End-to-End testing package.

Scope: Full API endpoint testing simulating real user requests.
Rules: Uses FastAPI's TestClient. Mocks external APIs as needed to ensure
deterministic results and avoid costs, but tests the full HTTP request/response cycle.
"""

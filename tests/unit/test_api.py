"""Tests for the application interfaces (FastAPI routers).

This module validates the interaction between the outside world and the
application logic within the Interfaces layer.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_enhance_endpoint_success(mocker):
    """Test the /api/v1/enhance endpoint success scenario.

    Given: A valid request to the enhancement endpoint.
    When: The use case execution is successful.
    Then: The API should return a 200 status code and a success message.
    """
    mock_execute = mocker.patch(
        "app.application.enhancement.GraphEnhancementUseCase.execute", new_callable=mocker.AsyncMock
    )
    response = client.post("/api/v1/enhance")

    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}

    # Assert that execute was called once
    mock_execute.assert_called_once()

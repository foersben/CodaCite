from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.application.enhancement.GraphEnhancementUseCase.execute", new_callable=AsyncMock)
def test_enhance_endpoint_success(mock_execute):
    """Test the /api/v1/enhance endpoint."""
    response = client.post("/api/v1/enhance")

    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}

    # Assert that execute was called once
    mock_execute.assert_called_once()

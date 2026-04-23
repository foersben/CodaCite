from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_enhance_endpoint_success(mocker):
    """Test the /api/v1/enhance endpoint."""
    mock_execute = mocker.patch(
        "app.application.enhancement.GraphEnhancementUseCase.execute", new_callable=mocker.AsyncMock
    )
    response = client.post("/api/v1/enhance")

    assert response.status_code == 200
    assert response.json() == {"message": "Graph communities generated successfully."}

    # Assert that execute was called once
    mock_execute.assert_called_once()

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.routes.search.SearchService")
def test_search_endpoint_returns_results(mock_service_cls):
    mock_service = MagicMock()
    mock_service.search.return_value = [
        {"title": "Test Paper", "url": "https://arxiv.org/abs/1", "snippet": "..."}
    ]
    mock_service_cls.return_value = mock_service

    response = client.post("/api/v1/search", json={"query": "gradient descent", "limit": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "gradient descent"
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Test Paper"
from unittest.mock import patch

from app.core.exceptions import UpstreamServiceError, ResearchServiceError


# Patch the class where the route looks it up, so ResearchService() never builds
# real Firecrawl/Gemini providers (which need API keys that CI doesn't have).
@patch("app.api.routes.answer.ResearchService")
def test_answer_endpoint_returns_502_on_upstream_failure(mock_service_cls, client):
    mock_service_cls.return_value.answer.side_effect = UpstreamServiceError("Firecrawl down")

    response = client.post("/api/v1/answer", json={"question": "test question here"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Firecrawl down"


@patch("app.api.routes.answer.ResearchService")
def test_answer_endpoint_returns_503_on_pipeline_failure(mock_service_cls, client):
    mock_service_cls.return_value.answer.side_effect = ResearchServiceError("graph exploded")

    response = client.post("/api/v1/answer", json={"question": "test question here"})

    assert response.status_code == 503
    assert response.json()["detail"] == "graph exploded"

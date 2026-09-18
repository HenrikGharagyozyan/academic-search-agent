from app.core.exceptions import UpstreamServiceError, ResearchServiceError


def test_answer_endpoint_returns_502_on_upstream_failure(client, mock_research_service):
    mock_research_service.answer.side_effect = UpstreamServiceError("Firecrawl down")

    response = client.post("/api/v1/answer", json={"question": "test question here"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Firecrawl down"


def test_answer_endpoint_returns_503_on_pipeline_failure(client, mock_research_service):
    mock_research_service.answer.side_effect = ResearchServiceError("graph exploded")

    response = client.post("/api/v1/answer", json={"question": "test question here"})

    assert response.status_code == 503
    assert response.json()["detail"] == "graph exploded"

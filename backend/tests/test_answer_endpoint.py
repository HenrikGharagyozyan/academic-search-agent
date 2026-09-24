from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.domain.answers import Answer


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


def test_answer_rejects_too_short_question(client, mock_research_service):
    response = client.post("/api/v1/answer", json={"question": "hi"})
    assert response.status_code == 422


def test_answer_rejects_too_long_question(client, mock_research_service):
    response = client.post("/api/v1/answer", json={"question": "x" * 501})
    assert response.status_code == 422


def test_answer_accepts_valid_question(client, mock_research_service):
    mock_research_service.answer.return_value = Answer(
        question="valid question here",
        summary="s",
        claims=[],
        conclusion="c",
        evidence={},
        evidence_sufficient=True,
    )

    response = client.post("/api/v1/answer", json={"question": "valid question here"})
    assert response.status_code == 200
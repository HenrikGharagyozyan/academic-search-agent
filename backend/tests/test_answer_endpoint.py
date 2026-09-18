from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.routes.answer.ResearchService")
def test_answer_rejects_too_short_question(mock_service_cls):
    response = client.post("/api/v1/answer", json={"question": "hi"})
    assert response.status_code == 422


@patch("app.api.routes.answer.ResearchService")
def test_answer_rejects_too_long_question(mock_service_cls):
    response = client.post("/api/v1/answer", json={"question": "x" * 501})
    assert response.status_code == 422


@patch("app.api.routes.answer.ResearchService")
def test_answer_accepts_valid_question(mock_service_cls):
    from app.schemas.answer import Answer

    mock_service_cls.return_value.answer.return_value = Answer(
        question="valid question here",
        summary="s", claims=[], conclusion="c", evidence={},
    )

    response = client.post("/api/v1/answer", json={"question": "valid question here"})
    assert response.status_code == 200
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.routes.documents.DocumentService")
def test_get_document_returns_parsed_document(mock_service_cls):
    from app.schemas.document import DocumentLine, ParsedDocument

    mock_service = MagicMock()
    mock_service.get_document.return_value = ParsedDocument(
        url="https://example.com",
        title="Example Page",
        lines=[DocumentLine(line_number=1, text="Some content")],
    )
    mock_service_cls.return_value = mock_service

    response = client.post("/api/v1/documents", json={"url": "https://example.com"})

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Example Page"
    assert data["lines"][0]["text"] == "Some content"

from app.schemas.document import DocumentLine, ParsedDocument


def test_get_document_returns_parsed_document(client, mock_document_service):
    mock_document_service.get_document.return_value = ParsedDocument(
        url="https://example.com",
        title="Example Page",
        lines=[DocumentLine(line_number=1, text="Some content")],
    )

    response = client.post("/api/v1/documents", json={"url": "https://example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Example Page"
    assert body["lines"][0]["text"] == "Some content"
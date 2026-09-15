from unittest.mock import MagicMock, patch

from app.providers.firecrawl_provider import FirecrawlProvider


@patch("app.providers.firecrawl_provider.FirecrawlApp")
def test_scrape_parses_page(mock_app_cls, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from firecrawl.v2.types import Document, DocumentMetadata

    mock_client = MagicMock()
    mock_client.scrape.return_value = Document(
        markdown="# Some content",
        metadata=DocumentMetadata(title="Test Page"),
    )
    mock_app_cls.return_value = mock_client

    provider = FirecrawlProvider()
    page = provider.scrape("https://example.com")

    assert page.title == "Test Page"
    assert page.markdown == "# Some content"
    assert page.url == "https://example.com"
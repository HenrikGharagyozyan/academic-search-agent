from unittest.mock import MagicMock, patch

from app.infrastructure.search.firecrawl import FirecrawlProvider


@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
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


@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
def test_scrape_uses_cache_on_second_call(mock_app_cls):
    provider = FirecrawlProvider()

    call_count = {"n": 0}

    class FakeMetadata:
        title = "Cached Title"

    class FakeResponse:
        metadata = FakeMetadata()
        markdown = "content"

    def fake_scrape(url, formats):
        call_count["n"] += 1
        return FakeResponse()

    provider._client.scrape = fake_scrape

    first = provider.scrape("https://example.com")
    second = provider.scrape("https://example.com")

    assert first == second
    assert call_count["n"] == 1
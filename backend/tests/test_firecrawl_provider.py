from unittest.mock import MagicMock, patch

import pytest

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

# --- a search refused for quota ---------------------------------------------


def _web(*urls):
    response = MagicMock()
    response.web = [MagicMock(title="t", url=url, description="d") for url in urls]
    return response


@patch("app.infrastructure.search.firecrawl.time.sleep")
@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
def test_a_rate_limited_search_is_retried_then_succeeds(mock_app_cls, sleep):
    """Searches share the scrapes' per-minute quota; a refused search used to
    drop one planned query without a second try."""
    client = mock_app_cls.return_value
    client.search.side_effect = [
        RuntimeError("Rate Limit Exceeded: Consumed (req/min): 11, Remaining (req/min): 0"),
        _web("https://arxiv.org/abs/1"),
    ]

    results = FirecrawlProvider().search("transmission matrix", limit=5)

    assert [r.url for r in results] == ["https://arxiv.org/abs/1"]
    assert client.search.call_count == 2
    sleep.assert_called_once()


@patch("app.infrastructure.search.firecrawl.time.sleep")
@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
def test_a_persistent_rate_limit_on_search_is_reported_as_itself(mock_app_cls, sleep):
    from app.infrastructure.search.firecrawl import RATE_LIMIT_RETRIES, SearchRateLimited

    client = mock_app_cls.return_value
    client.search.side_effect = RuntimeError("429 Too Many Requests")

    with pytest.raises(SearchRateLimited):
        FirecrawlProvider().search("q")

    assert client.search.call_count == RATE_LIMIT_RETRIES + 1


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("connection reset"),
        # Out of credits is not a per-minute limit: waiting does not refill it.
        RuntimeError("Payment Required: Failed to search. Insufficient credits"),
    ],
)
@patch("app.infrastructure.search.firecrawl.time.sleep")
@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
def test_a_search_error_that_is_not_a_rate_limit_is_not_retried(mock_app_cls, sleep, error):
    client = mock_app_cls.return_value
    client.search.side_effect = error

    with pytest.raises(RuntimeError):
        FirecrawlProvider().search("q")

    assert client.search.call_count == 1
    sleep.assert_not_called()


# --- the page's own citation record -----------------------------------------


@patch("app.infrastructure.search.firecrawl.FirecrawlApp")
def test_scrape_reads_the_citation_tags(mock_app_cls):
    from firecrawl.v2.types import Document, DocumentMetadata

    mock_app_cls.return_value.scrape.return_value = Document(
        markdown="# Paper",
        metadata=DocumentMetadata(
            title="Wavefront shaping",
            citation_author=["Resisi, Shachar", "Bromberg, Yaron"],
            citation_date="2019/10/07",
        ),
    )

    page = FirecrawlProvider().scrape("https://arxiv.org/abs/1910.02798")

    assert page.citation.year == 2019
    assert page.citation.authors == ["Resisi, Shachar", "Bromberg, Yaron"]

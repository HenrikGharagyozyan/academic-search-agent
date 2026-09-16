from unittest.mock import MagicMock

from app.agents.nodes import generate_claims_node, retrieve_and_chunk_node, search_node
from app.providers.firecrawl_provider import ScrapedPage, SearchResult
from app.schemas.answer import Claim


def test_search_node_calls_firecrawl_search():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]

    result = search_node(
        {"question": "test question", "search_query": "test question"},
        firecrawl=mock_firecrawl,
    )

    assert len(result["search_results"]) == 1
    mock_firecrawl.search.assert_called_once_with("test question", limit=3)


def test_retrieve_and_chunk_node_skips_failed_scrape():
    mock_firecrawl = MagicMock()

    def fake_scrape(url):
        if url == "https://broken.com":
            raise RuntimeError("boom")
        return ScrapedPage(url=url, title="OK", markdown="Some content here.")

    mock_firecrawl.scrape.side_effect = fake_scrape

    state = {
        "search_results": [
            SearchResult(title="Broken", url="https://broken.com", snippet="..."),
            SearchResult(title="Working", url="https://working.com", snippet="..."),
        ]
    }

    result = retrieve_and_chunk_node(state, firecrawl=mock_firecrawl)

    assert len(result["chunks"]) > 0
    assert all(c.source_url == "https://working.com" for c in result["chunks"])


def test_generate_claims_node_returns_empty_when_no_chunks():
    mock_gemini = MagicMock()

    result = generate_claims_node({"chunks": []}, gemini=mock_gemini)

    assert result["claims"] == []
    mock_gemini.generate_claims.assert_not_called()
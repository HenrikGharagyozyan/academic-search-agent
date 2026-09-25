from unittest.mock import MagicMock
import pytest

from app.application.agents.constants import MAX_SOURCES
from app.application.agents.nodes import generate_claims_node, retrieve_and_chunk_node, search_node
from app.domain.search import ScrapedPage, SearchResult
from app.core.exceptions import UpstreamServiceError


def test_search_node_calls_firecrawl_search():
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]

    result = search_node(
        {"question": "test question", "search_query": "test question"},
        search_provider=mock_search,
    )

    assert len(result["search_results"]) == 1
    mock_search.search.assert_called_once_with("test question", limit=MAX_SOURCES)


def test_retrieve_and_chunk_node_skips_failed_scrape():
    mock_search = MagicMock()

    def fake_scrape(url):
        if url == "https://broken.com":
            raise RuntimeError("boom")
        return ScrapedPage(url=url, title="OK", markdown="Some content here.")

    mock_search.scrape.side_effect = fake_scrape

    state = {
        "search_results": [
            SearchResult(title="Broken", url="https://broken.com", snippet="..."),
            SearchResult(title="Working", url="https://working.com", snippet="..."),
        ]
    }

    result = retrieve_and_chunk_node(state, search_provider=mock_search)

    assert len(result["chunks"]) > 0
    assert all(c.source_url == "https://working.com" for c in result["chunks"])


def test_generate_claims_node_returns_empty_when_no_chunks():
    mock_llm = MagicMock()

    result = generate_claims_node({"selected_chunks": []}, llm=mock_llm)

    assert result["summary"] == ""
    assert result["claims"] == []
    assert result["conclusion"] == ""
    mock_llm.generate_answer.assert_not_called()


def test_select_relevant_chunks_node_calls_vector_store():
    from app.application.agents.nodes import select_relevant_chunks_node
    from app.domain.documents import Chunk

    mock_vector_store = MagicMock()
    chunk = Chunk(
        chunk_id="c1", document_id="doc1", text="text",
        start_line=1, end_line=1, source_url="https://x.com", title="X",
    )
    mock_vector_store.select_relevant_chunks.return_value = [chunk]

    state = {"question": "q?", "chunks": [chunk]}
    result = select_relevant_chunks_node(state, vector_store=mock_vector_store)

    assert result["selected_chunks"] == [chunk]


def test_generate_claims_node_returns_empty_on_gemini_failure():
    from app.application.agents.nodes import generate_claims_node
    from app.domain.documents import Chunk

    mock_llm = MagicMock()
    mock_llm.generate_answer.side_effect = RuntimeError("503 UNAVAILABLE")

    chunk = Chunk(
        chunk_id="c1", document_id="doc1", text="text",
        start_line=1, end_line=1, source_url="https://x.com", title="X",
    )
    state = {"question": "q?", "selected_chunks": [chunk]}

    result = generate_claims_node(state, llm=mock_llm)

    assert result["summary"] == ""
    assert result["claims"] == []
    assert result["conclusion"] == ""


def test_search_node_raises_upstream_error_on_firecrawl_failure():
    mock_search = MagicMock()
    mock_search.search.side_effect = RuntimeError("boom")

    with pytest.raises(UpstreamServiceError):
        search_node(
            {"question": "test question", "search_query": "test question"},
            search_provider=mock_search,
        )
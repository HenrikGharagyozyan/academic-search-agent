from unittest.mock import MagicMock

from app.agents.graph import build_research_graph
from app.providers.firecrawl_provider import ScrapedPage, SearchResult
from app.schemas.answer import Claim


def test_graph_runs_end_to_end_with_mocks():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()

    def fake_generate_claims(question, evidence_chunks):
        return [
            Claim(
                text="A claim",
                evidence_ids=[evidence_chunks[0]["chunk_id"]],
                confidence="high",
            )
        ]

    mock_gemini.generate_claims.side_effect = fake_generate_claims

    graph = build_research_graph(firecrawl=mock_firecrawl, gemini=mock_gemini)
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "claims": [],
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert len(result["chunks"]) > 0
    assert len(result["claims"]) == 1


def test_graph_retries_when_no_evidence_found_then_succeeds():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.refine_query.return_value = "refined query"

    call_count = {"n": 0}

    def fake_generate_claims(question, evidence_chunks):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [Claim(text="Hallucinated", evidence_ids=["nonexistent"], confidence="low")]
        return [
            Claim(
                text="Grounded claim",
                evidence_ids=[evidence_chunks[0]["chunk_id"]],
                confidence="high",
            )
        ]

    mock_gemini.generate_claims.side_effect = fake_generate_claims

    graph = build_research_graph(firecrawl=mock_firecrawl, gemini=mock_gemini)
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "claims": [],
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert result["evidence_sufficient"] is True
    assert result["retry_count"] == 1
    mock_gemini.refine_query.assert_called_once()


def test_graph_stops_after_max_retries_with_no_evidence():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.refine_query.return_value = "refined query"
    mock_gemini.generate_claims.return_value = [
        Claim(text="Hallucinated", evidence_ids=["nonexistent"], confidence="low")
    ]

    graph = build_research_graph(firecrawl=mock_firecrawl, gemini=mock_gemini)
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "claims": [],
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert result["evidence_sufficient"] is False
    assert result["retry_count"] == 2  # MAX_RETRIES
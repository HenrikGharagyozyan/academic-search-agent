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
        {"question": "test?", "search_results": [], "chunks": [], "claims": []}
    )

    assert len(result["chunks"]) > 0
    assert len(result["claims"]) == 1
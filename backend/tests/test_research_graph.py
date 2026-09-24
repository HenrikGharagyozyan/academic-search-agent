from unittest.mock import MagicMock

from app.agents.constants import MAX_RETRIES
from app.agents.graph import build_research_graph
from app.domain.search import ScrapedPage, SearchResult
from app.domain.answers import Claim, ClaimsResponse


def test_graph_runs_end_to_end_with_mocks(keep_all_chunks_relevant):
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.side_effect = keep_all_chunks_relevant

    def fake_generate_answer(question, evidence_chunks):
        return ClaimsResponse(
            summary="Test summary",
            claims=[
                Claim(
                    text="A claim",
                    evidence_ids=[evidence_chunks[0]["chunk_id"]],
                    confidence="high",
                )
            ],
            conclusion="Test conclusion",
        )

    mock_gemini.generate_answer.side_effect = fake_generate_answer

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    graph = build_research_graph(firecrawl=mock_firecrawl, gemini=mock_gemini, vector_store=mock_vector_store)
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "summary": "",
            "claims": [],
            "conclusion": "",
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert len(result["chunks"]) > 0
    assert len(result["claims"]) == 1


def test_graph_retries_when_no_evidence_found_then_succeeds(keep_all_chunks_relevant):
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.side_effect = keep_all_chunks_relevant
    mock_gemini.refine_query.return_value = "refined query"

    call_count = {"n": 0}

    def fake_generate_answer(question, evidence_chunks):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ClaimsResponse(
                summary="Ungrounded summary",
                claims=[
                    Claim(text="Hallucinated", evidence_ids=["nonexistent"], confidence="low")
                ],
                conclusion="Ungrounded conclusion",
            )
        return ClaimsResponse(
            summary="Test summary",
            claims=[
                Claim(
                    text="Grounded claim",
                    evidence_ids=[evidence_chunks[0]["chunk_id"]],
                    confidence="high",
                )
            ],
            conclusion="Test conclusion",
        )

    mock_gemini.generate_answer.side_effect = fake_generate_answer

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    graph = build_research_graph(
        firecrawl=mock_firecrawl, gemini=mock_gemini, vector_store=mock_vector_store
    )
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "summary": "",
            "claims": [],
            "conclusion": "",
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert result["evidence_sufficient"] is True
    assert result["retry_count"] == 1
    mock_gemini.refine_query.assert_called_once()


def test_graph_stops_after_max_retries_with_no_evidence(keep_all_chunks_relevant):
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.side_effect = keep_all_chunks_relevant
    mock_gemini.refine_query.return_value = "refined query"
    mock_gemini.generate_answer.return_value = ClaimsResponse(
        summary="Ungrounded summary",
        claims=[Claim(text="Hallucinated", evidence_ids=["nonexistent"], confidence="low")],
        conclusion="Ungrounded conclusion",
    )

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    graph = build_research_graph(
        firecrawl=mock_firecrawl, gemini=mock_gemini, vector_store=mock_vector_store
    )
    result = graph.invoke(
        {
            "question": "test?",
            "search_query": "test?",
            "search_results": [],
            "chunks": [],
            "selected_chunks": [],
            "summary": "",
            "claims": [],
            "conclusion": "",
            "retry_count": 0,
            "evidence_sufficient": False,
        }
    )

    assert result["evidence_sufficient"] is False
    assert result["retry_count"] == MAX_RETRIES
from unittest.mock import MagicMock

from app.application.agents.constants import MAX_RETRIES
from app.domain.search import ScrapedPage, SearchResult
from app.domain.answers import Claim, ClaimsResponse
from app.application.services.research import ResearchService


def test_answer_builds_evidence_from_used_claims(keep_all_chunks_relevant):
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_search.scrape.return_value = ScrapedPage(
        url="https://example.com",
        title="Paper Title",
        markdown="Gradient descent converges for convex functions.",
    )

    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = keep_all_chunks_relevant

    def fake_generate_answer(question, evidence_chunks):
        first_id = evidence_chunks[0].chunk_id
        return ClaimsResponse(
            summary="Test summary",
            claims=[
                Claim(
                    text="Gradient descent converges for convex functions.",
                    evidence_ids=[first_id],
                    confidence="high",
                )
            ],
            conclusion="Test conclusion",
        )

    mock_llm.generate_answer.side_effect = fake_generate_answer

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    service = ResearchService(search_provider=mock_search, llm=mock_llm, vector_store=mock_vector_store)
    result = service.answer("Does gradient descent converge?")

    assert result.summary == "Test summary"
    assert result.conclusion == "Test conclusion"
    assert len(result.claims) == 1
    assert len(result.evidence) == 1
    evidence_item = list(result.evidence.values())[0]
    assert evidence_item.source_url == "https://example.com"
    assert evidence_item.title == "Paper Title"
    assert result.evidence_sufficient is True


def test_answer_drops_claims_with_only_unknown_evidence_ids(keep_all_chunks_relevant):
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_search.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = keep_all_chunks_relevant
    mock_llm.generate_answer.return_value = ClaimsResponse(
        summary="Test summary",
        claims=[
            Claim(text="Hallucinated claim", evidence_ids=["nonexistent_id"], confidence="low")
        ],
        conclusion="Test conclusion",
    )
    mock_llm.refine_query.return_value = "refined query"

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    service = ResearchService(
        search_provider=mock_search, llm=mock_llm, vector_store=mock_vector_store
    )
    result = service.answer("Some question?")

    # verify_evidence strips the ungrounded claim, so no evidence survives either
    assert result.claims == []
    assert result.evidence == {}
    # insufficient evidence triggers the refine loop until MAX_RETRIES is hit
    assert mock_llm.generate_answer.call_count == MAX_RETRIES + 1

def test_answer_skips_failed_scrape_and_continues(keep_all_chunks_relevant):
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(title="Broken", url="https://broken.com", snippet="..."),
        SearchResult(title="Working", url="https://working.com", snippet="..."),
    ]

    def fake_scrape(url):
        if url == "https://broken.com":
            raise RuntimeError("Website Not Supported")
        return ScrapedPage(
            url=url, title="Working Page", markdown="Some working content here."
        )

    mock_search.scrape.side_effect = fake_scrape

    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = keep_all_chunks_relevant

    def fake_generate_answer(question, evidence_chunks):
        first_id = evidence_chunks[0].chunk_id
        return ClaimsResponse(
            summary="Test summary",
            claims=[Claim(text="Some claim", evidence_ids=[first_id], confidence="high")],
            conclusion="Test conclusion",
        )

    mock_llm.generate_answer.side_effect = fake_generate_answer

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    service = ResearchService(
        search_provider=mock_search, llm=mock_llm, vector_store=mock_vector_store
    )
    result = service.answer("Some question?")

    assert len(result.claims) == 1
    assert len(result.evidence) == 1
    assert list(result.evidence.values())[0].source_url == "https://working.com"
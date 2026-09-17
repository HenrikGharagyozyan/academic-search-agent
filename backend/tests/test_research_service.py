from unittest.mock import MagicMock

from app.providers.firecrawl_provider import ScrapedPage, SearchResult
from app.schemas.answer import Claim
from app.services.research_service import ResearchService


def test_answer_builds_evidence_from_used_claims():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com",
        title="Paper Title",
        markdown="Gradient descent converges for convex functions.",
    )

    mock_gemini = MagicMock()

    def fake_generate_claims(question, evidence_chunks):
        first_id = evidence_chunks[0]["chunk_id"]
        return [
            Claim(
                text="Gradient descent converges for convex functions.",
                evidence_ids=[first_id],
                confidence="high",
            )
        ]

    mock_gemini.generate_claims.side_effect = fake_generate_claims

    mock_vector_store = MagicMock()
    mock_vector_store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    service = ResearchService(firecrawl=mock_firecrawl, gemini=mock_gemini, vector_store=mock_vector_store)
    result = service.answer("Does gradient descent converge?")

    assert len(result.claims) == 1
    assert len(result.evidence) == 1
    evidence_item = list(result.evidence.values())[0]
    assert evidence_item.source_url == "https://example.com"
    assert evidence_item.title == "Paper Title"


def test_answer_drops_evidence_for_unknown_ids():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    mock_firecrawl.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Some content here."
    )

    mock_gemini = MagicMock()
    mock_gemini.generate_claims.return_value = [
        Claim(text="Hallucinated claim", evidence_ids=["nonexistent_id"], confidence="low")
    ]

    service = ResearchService(firecrawl=mock_firecrawl, gemini=mock_gemini)
    result = service.answer("Some question?")

    assert len(result.claims) == 1
    assert len(result.evidence) == 0  # fake id in evidence


def test_answer_skips_failed_scrape_and_continues():
    mock_firecrawl = MagicMock()
    mock_firecrawl.search.return_value = [
        SearchResult(title="Broken", url="https://broken.com", snippet="..."),
        SearchResult(title="Working", url="https://working.com", snippet="..."),
    ]

    def fake_scrape(url):
        if url == "https://broken.com":
            raise RuntimeError("Website Not Supported")
        return ScrapedPage(
            url=url, title="Working Page", markdown="Some working content here."
        )

    mock_firecrawl.scrape.side_effect = fake_scrape

    mock_gemini = MagicMock()

    def fake_generate_claims(question, evidence_chunks):
        first_id = evidence_chunks[0]["chunk_id"]
        return [Claim(text="Some claim", evidence_ids=[first_id], confidence="high")]

    mock_gemini.generate_claims.side_effect = fake_generate_claims

    service = ResearchService(firecrawl=mock_firecrawl, gemini=mock_gemini)
    result = service.answer("Some question?")

    assert len(result.claims) == 1
    assert len(result.evidence) == 1
    assert list(result.evidence.values())[0].source_url == "https://working.com"
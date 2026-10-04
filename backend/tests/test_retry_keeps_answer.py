"""A retry can only add to an answer, never take one away.

The first pass found grounded claims and the grader called them thin, so the
graph searched again. When that second pass found nothing — the pages could
not be read, or every search failed — the run ended with the second pass's
empty state, and the reader got no answer or an error in place of a thin one.
"""

from unittest.mock import MagicMock

from app.application.services.research import ResearchService
from app.domain.answers import Claim, ClaimsResponse
from app.domain.grading import AnswerQualityGrade
from app.domain.search import ScrapedPage, SearchResult

THIN = AnswerQualityGrade(
    is_satisfactory=False, reasoning="one source only", problem="too_thin", missing="reviews"
)


def _service(second_search):
    """A pipeline whose first pass answers thinly and whose second pass is
    whatever ``second_search`` makes of it."""
    search = MagicMock()

    def run_search(query, limit, since_year):
        if query == "refined query":
            return second_search()
        return [SearchResult(title="Paper", url="https://example.com/a", snippet="...")]

    search.search.side_effect = run_search
    search.scrape.return_value = ScrapedPage(
        url="https://example.com/a", title="Paper", markdown="Grounded content about the topic."
    )

    llm = MagicMock()
    llm.plan_searches.side_effect = RuntimeError("plan the question as-is")
    llm.grade_relevance.side_effect = lambda q, chunks: MagicMock(
        relevant_chunk_ids=[c.chunk_id for c in chunks], reasoning=""
    )
    llm.grade_answer_quality.return_value = THIN
    llm.refine_query.return_value = "refined query"

    def generate(question, chunks):
        return ClaimsResponse(
            summary="Thin but grounded.",
            claims=[Claim(text="A grounded claim.", evidence_ids=[chunks[0].chunk_id], confidence="medium")],
            conclusion="Only one source.",
        )

    llm.generate_answer.side_effect = generate

    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks
    return ResearchService(search_provider=search, llm=llm, vector_store=store)


def test_a_retry_that_finds_nothing_keeps_the_first_answer():
    answer = _service(second_search=lambda: []).answer("q?")

    assert [c.text for c in answer.claims] == ["A grounded claim."]
    assert answer.evidence, "the claim's passage must still be there to cite"


def test_a_retry_whose_searches_all_fail_keeps_the_first_answer():
    def outage():
        raise RuntimeError("provider down")

    answer = _service(second_search=outage).answer("q?")

    assert [c.text for c in answer.claims] == ["A grounded claim."]

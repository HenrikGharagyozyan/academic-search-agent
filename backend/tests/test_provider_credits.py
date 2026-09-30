"""An empty balance is reported, not dressed up as "no reliable sources".

OpenRouter refused every call of a run with 402. Each node treats a failed
model call as a degraded step and carries on, so grading kept everything,
generation returned nothing, and the reader was told to rephrase the question.
"""

from unittest.mock import MagicMock

import pytest

from app.application.agents.nodes import generate_claims_node, grade_relevance_node
from app.application.services.research import ResearchService
from app.core.exceptions import ProviderCreditsExhausted
from app.domain.answers import ClaimsResponse
from app.domain.documents import Chunk
from app.domain.search import ScrapedPage, SearchResult
from app.infrastructure.llm.base import LangChainLLMProvider
from app.infrastructure.llm.retry import is_billing_error, is_transient_error

# Verbatim from the backend log.
REFUSED = RuntimeError(
    "Error code: 402 - {'error': {'message': \"This request requires more credits, "
    "or fewer max_tokens. You requested up to 8192 tokens, but can only afford 5605.\""
)
IN_FLIGHT = RuntimeError(
    "Error code: 402 - {'error': {'message': 'This request would exceed your available "
    "credits given your current in-flight requests. Retry after in-flight requests settle"
)


def chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc", text="passage",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


def test_a_refusal_to_pay_is_a_billing_error_and_not_retried():
    assert is_billing_error(REFUSED)
    assert not is_transient_error(REFUSED)


def test_a_refusal_over_requests_in_flight_is_waited_out():
    assert not is_billing_error(IN_FLIGHT)
    assert is_transient_error(IN_FLIGHT)


def test_the_provider_raises_a_named_error_after_one_attempt():
    llm = MagicMock()
    structured = llm.with_structured_output.return_value
    structured.invoke.side_effect = REFUSED
    provider = LangChainLLMProvider(llm, provider_name="openrouter", model_name="m")

    with pytest.raises(ProviderCreditsExhausted, match="openrouter.*out of credits"):
        provider.generate_answer("q?", [chunk()])

    assert structured.invoke.call_count == 1


def test_generation_does_not_swallow_it():
    llm = MagicMock()
    llm.generate_answer.side_effect = ProviderCreditsExhausted("out of credits")

    with pytest.raises(ProviderCreditsExhausted):
        generate_claims_node({"question": "q?", "selected_chunks": [chunk()]}, llm=llm)


def test_relevance_grading_does_not_keep_every_passage_instead():
    llm = MagicMock()
    llm.grade_relevance.side_effect = ProviderCreditsExhausted("out of credits")

    with pytest.raises(ProviderCreditsExhausted):
        grade_relevance_node({"question": "q?", "selected_chunks": [chunk()]}, llm=llm)


def test_the_reader_is_told_the_provider_is_out_of_credits(pipeline_with_llm):
    service, llm = pipeline_with_llm
    llm.generate_answer.side_effect = ProviderCreditsExhausted(
        "The language model provider (openrouter) is out of credits"
    )

    events = list(service.stream_answer("Does gradient descent converge?"))

    assert events[-1]["event"] == "error"
    assert "out of credits" in events[-1]["data"]["detail"]


@pytest.fixture
def pipeline_with_llm(keep_all_chunks_relevant, answer_is_satisfactory):
    search = MagicMock()
    search.search.return_value = [SearchResult(title="P", url="https://example.com", snippet="...")]
    search.scrape.return_value = ScrapedPage(
        url="https://example.com", title="P", markdown="Gradient descent converges."
    )
    llm = MagicMock()
    llm.grade_relevance.side_effect = keep_all_chunks_relevant
    llm.grade_answer_quality.return_value = answer_is_satisfactory
    llm.generate_answer.return_value = ClaimsResponse(summary="", claims=[], conclusion="")
    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks
    return ResearchService(search_provider=search, llm=llm, vector_store=store), llm

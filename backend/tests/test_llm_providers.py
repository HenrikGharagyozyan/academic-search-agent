"""Both vendor adapters go through one shared base, so these tests run the
same expectations against each of them. The drift they guard against is real:
the two providers were once separate copies, and one of them read
``AIMessage.content`` — which is a list for models that return content blocks —
while the other read ``.text``.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import AnswerQualityGrade, RelevanceGrade
from app.infrastructure.llm.base import LangChainLLMProvider
from app.infrastructure.llm.retry import MAX_ATTEMPTS, is_transient_error


def make_chunk(chunk_id: str = "ev_1", text: str = "some evidence") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", text=text,
        start_line=1, end_line=1, source_url="https://example.com", title="Example",
    )


def build_provider(llm: MagicMock) -> LangChainLLMProvider:
    return LangChainLLMProvider(llm, provider_name="test", model_name="test-model")


def structured_llm(llm: MagicMock) -> MagicMock:
    """The base builds one structured runnable per output schema up front."""
    return llm.with_structured_output.return_value


@pytest.fixture
def llm() -> MagicMock:
    return MagicMock()


def test_generate_answer_passes_chunks_as_an_evidence_block(llm):
    structured_llm(llm).invoke.return_value = ClaimsResponse(
        summary="s", claims=[Claim(text="c", evidence_ids=["ev_1"], confidence="high")],
        conclusion="c",
    )

    result = build_provider(llm).generate_answer("q?", [make_chunk("ev_1", "body text")])

    assert result.summary == "s"
    prompt = structured_llm(llm).invoke.call_args.args[0]
    rendered = str(prompt)
    assert "ev_1" in rendered and "body text" in rendered


def test_refine_query_reads_text_not_content(llm):
    # A model returning content blocks makes `.content` a list; `.text` flattens both.
    llm.invoke.return_value = AIMessage(content=[{"type": "text", "text": "  refined  "}])

    assert build_provider(llm).refine_query("q?", "old query") == "refined"


def test_grade_relevance_returns_the_structured_grade(llm):
    structured_llm(llm).invoke.return_value = RelevanceGrade(
        relevant_chunk_ids=["ev_1"], reasoning="on topic"
    )

    grade = build_provider(llm).grade_relevance("q?", [make_chunk()])

    assert grade.relevant_chunk_ids == ["ev_1"]


def test_grade_answer_quality_returns_the_structured_grade(llm):
    structured_llm(llm).invoke.return_value = AnswerQualityGrade(
        is_satisfactory=False, reasoning="off topic"
    )

    grade = build_provider(llm).grade_answer_quality(
        "q?", "summary", [Claim(text="c", evidence_ids=["ev_1"], confidence="low")], "conclusion"
    )

    assert grade.is_satisfactory is False


def test_transient_failure_is_retried_then_succeeds(llm):
    structured_llm(llm).invoke.side_effect = [
        RuntimeError("503 UNAVAILABLE"),
        ClaimsResponse(summary="s", claims=[], conclusion="c"),
    ]

    result = build_provider(llm).generate_answer("q?", [make_chunk()])

    assert result.summary == "s"
    assert structured_llm(llm).invoke.call_count == 2


def test_transient_failure_gives_up_after_max_attempts(llm):
    structured_llm(llm).invoke.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError):
        build_provider(llm).generate_answer("q?", [make_chunk()])

    assert structured_llm(llm).invoke.call_count == MAX_ATTEMPTS


def test_permanent_failure_is_not_retried(llm):
    structured_llm(llm).invoke.side_effect = ValueError("malformed request")

    with pytest.raises(ValueError):
        build_provider(llm).generate_answer("q?", [make_chunk()])

    assert structured_llm(llm).invoke.call_count == 1


@pytest.mark.parametrize(
    "message",
    ["429 too many requests", "503 Service Unavailable", "RESOURCE_EXHAUSTED",
     "model is overloaded", "502 Bad Gateway", "request timed out"],
)
def test_every_vendor_wording_for_a_transient_failure_is_recognised(message):
    # Each marker came from one vendor's error text; the policy is shared, so
    # all of them must be honoured for every provider.
    assert is_transient_error(RuntimeError(message))


def test_a_genuine_error_is_not_mistaken_for_a_transient_one():
    assert not is_transient_error(ValueError("invalid evidence id"))

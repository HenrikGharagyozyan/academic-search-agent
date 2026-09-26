"""Both vendor adapters go through one shared base, so these tests run the
same expectations against each of them. The drift they guard against is real:
the two providers were once separate copies, and one of them read
``AIMessage.content`` — which is a list for models that return content blocks —
while the other read ``.text``.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import AnswerQualityGrade, RelevanceGrade
from app.infrastructure.llm.base import LangChainLLMProvider
from app.infrastructure.llm.registry import (
    REGISTRY,
    available_providers,
    create_llm_provider,
    resolve_model,
)
from app.infrastructure.llm.retry import MAX_ATTEMPTS, is_transient_error
from app.ports.llm import LLMProvider


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


# --- registry -------------------------------------------------------------


def test_registry_covers_every_configurable_provider_name():
    # Settings validates LLM_PROVIDER against a Literal; the registry must be
    # able to build each name that Literal admits, or config passes and the
    # build blows up at request time.
    allowed = set(Settings.model_fields["llm_provider"].annotation.__args__)
    assert allowed == set(REGISTRY)
    assert available_providers() == sorted(allowed)


@pytest.mark.parametrize("provider_name", sorted(REGISTRY))
def test_each_registered_provider_satisfies_the_port(provider_name):
    spec = REGISTRY[provider_name]
    assert isinstance(spec.default_model, str) and spec.default_model
    assert issubclass(spec.build, LangChainLLMProvider)


def test_resolve_model_prefers_the_configured_override():
    settings = Settings(
        _env_file=None, firecrawl_api_key="x", gemini_api_key="x",
        llm_provider="gemini", llm_model="gemini-custom",
    )
    assert resolve_model(settings) == "gemini-custom"


def test_resolve_model_falls_back_to_the_provider_default():
    settings = Settings(
        _env_file=None, firecrawl_api_key="x", gemini_api_key="x", llm_provider="gemini",
    )
    assert resolve_model(settings) == REGISTRY["gemini"].default_model


def test_create_llm_provider_builds_the_configured_provider(monkeypatch):
    built = {}

    def fake_chat_openai(**kwargs):
        built.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("app.infrastructure.llm.openrouter.ChatOpenAI", fake_chat_openai)
    settings = Settings(
        _env_file=None, firecrawl_api_key="x", gemini_api_key="x",
        openrouter_api_key="or-key", llm_provider="openrouter", llm_model="some/model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, LLMProvider)
    assert provider.provider_name == "openrouter"
    assert provider.model_name == "some/model"
    assert built["model"] == "some/model"
    assert built["api_key"] == "or-key"

from unittest.mock import MagicMock, patch

import pytest

from app.domain.answers import Claim, ClaimsResponse


@patch("app.providers.gemini_provider.ChatGoogleGenerativeAI")
def test_generate_answer_retries_on_transient_error_then_succeeds(mock_llm_cls, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from app.providers.gemini_provider import GeminiProvider

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = [
        RuntimeError("503 UNAVAILABLE"),
        ClaimsResponse(
            summary="Test summary",
            claims=[Claim(text="ok", evidence_ids=["ev_1"], confidence="high")],
            conclusion="Test conclusion",
        ),
    ]

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    provider = GeminiProvider()
    result = provider.generate_answer("q?", [{"chunk_id": "ev_1", "text": "x"}])

    assert result.summary == "Test summary"
    assert len(result.claims) == 1
    assert result.conclusion == "Test conclusion"
    assert mock_structured_llm.invoke.call_count == 2


@patch("app.providers.gemini_provider.ChatGoogleGenerativeAI")
def test_generate_answer_gives_up_after_max_attempts(mock_llm_cls, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from app.providers.gemini_provider import GeminiProvider

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    provider = GeminiProvider()

    with pytest.raises(RuntimeError):
        provider.generate_answer("q?", [{"chunk_id": "ev_1", "text": "x"}])

    assert mock_structured_llm.invoke.call_count == 3


@patch("app.providers.gemini_provider.ChatGoogleGenerativeAI")
def test_refine_query_returns_stripped_text(mock_llm_cls, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from app.providers.gemini_provider import GeminiProvider

    mock_response = MagicMock()
    mock_response.text = "  refined query text  "

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_llm_cls.return_value = mock_llm_instance

    provider = GeminiProvider()
    result = provider.refine_query("original question", "old query")

    assert result == "refined query text"
    mock_llm_instance.invoke.assert_called_once()
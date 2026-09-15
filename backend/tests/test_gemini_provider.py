from unittest.mock import MagicMock, patch

from app.schemas.answer import Claim, ClaimsResponse


@patch("app.providers.gemini_provider.ChatGoogleGenerativeAI")
def test_generate_claims_parses_response(mock_llm_cls, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from app.providers.gemini_provider import GeminiProvider

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = ClaimsResponse(
        claims=[Claim(text="Test claim", evidence_ids=["ev_1"], confidence="high")]
    )

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_cls.return_value = mock_llm_instance

    provider = GeminiProvider()
    claims = provider.generate_claims(
        "What is X?", [{"chunk_id": "ev_1", "text": "X is Y"}]
    )

    assert len(claims) == 1
    assert claims[0].text == "Test claim"
    assert claims[0].evidence_ids == ["ev_1"]
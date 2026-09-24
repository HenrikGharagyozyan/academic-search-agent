from unittest.mock import MagicMock

from app.application.agents.nodes import verify_evidence_node
from app.application.agents.nodes import refine_query_node
from app.domain.answers import Claim
from app.domain.documents import Chunk


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", text="text",
        start_line=1, end_line=1, source_url="https://example.com", title="Example",
    )


def test_verify_evidence_sufficient_when_claim_grounded():
    state = {
        "selected_chunks": [make_chunk("ev_1")],
        "claims": [Claim(text="X", evidence_ids=["ev_1"], confidence="high")],
    }
    result = verify_evidence_node(state)

    assert result["evidence_sufficient"] is True
    assert result["claims"][0].evidence_ids == ["ev_1"]


def test_verify_evidence_strips_partially_hallucinated_ids():
    state = {
        "selected_chunks": [make_chunk("ev_1")],
        "claims": [Claim(text="X", evidence_ids=["ev_1", "fake_id"], confidence="high")],
    }
    result = verify_evidence_node(state)

    assert len(result["claims"]) == 1
    assert result["claims"][0].evidence_ids == ["ev_1"]


def test_verify_evidence_drops_fully_hallucinated_claim():
    state = {
        "selected_chunks": [make_chunk("ev_1")],
        "claims": [Claim(text="Y", evidence_ids=["fake_id"], confidence="low")],
    }
    result = verify_evidence_node(state)

    assert result["claims"] == []
    assert result["evidence_sufficient"] is False


def test_verify_evidence_insufficient_when_no_claims():
    state = {"selected_chunks": [make_chunk("ev_1")], "claims": []}
    result = verify_evidence_node(state)

    assert result["evidence_sufficient"] is False


def test_refine_query_node_returns_new_query_and_increments_retry():
    mock_gemini = MagicMock()
    mock_gemini.refine_query.return_value = "refined query"

    state = {"question": "q?", "search_query": "old query", "retry_count": 0}
    result = refine_query_node(state, gemini=mock_gemini)

    assert result["search_query"] == "refined query"
    assert result["retry_count"] == 1


def test_refine_query_node_keeps_previous_query_on_gemini_failure():
    mock_gemini = MagicMock()
    mock_gemini.refine_query.side_effect = RuntimeError("boom")

    state = {"question": "q?", "search_query": "old query", "retry_count": 1}
    result = refine_query_node(state, gemini=mock_gemini)

    assert result["search_query"] == "old query"
    assert result["retry_count"] == 2
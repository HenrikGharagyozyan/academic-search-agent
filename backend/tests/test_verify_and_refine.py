from unittest.mock import MagicMock

from app.agents.nodes import refine_query_node, verify_evidence_node
from app.schemas.answer import Claim
from app.schemas.document import Chunk


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text="some text",
        start_line=1,
        end_line=1,
        source_url="https://example.com",
        title="Example",
    )


def test_verify_evidence_sufficient_when_claim_grounded():
    state = {
        "chunks": [make_chunk("ev_1")],
        "claims": [Claim(text="X", evidence_ids=["ev_1"], confidence="high")],
    }

    result = verify_evidence_node(state)

    assert result["evidence_sufficient"] is True


def test_verify_evidence_insufficient_when_no_claims():
    state = {"chunks": [make_chunk("ev_1")], "claims": []}

    result = verify_evidence_node(state)

    assert result["evidence_sufficient"] is False


def test_verify_evidence_insufficient_when_evidence_ids_dont_exist():
    state = {
        "chunks": [make_chunk("ev_1")],
        "claims": [Claim(text="X", evidence_ids=["nonexistent"], confidence="low")],
    }

    result = verify_evidence_node(state)

    assert result["evidence_sufficient"] is False


def test_refine_query_node_increments_retry_and_calls_gemini():
    mock_gemini = MagicMock()
    mock_gemini.refine_query.return_value = "better search query"

    state = {
        "question": "original question",
        "search_query": "old query",
        "retry_count": 0,
    }

    result = refine_query_node(state, gemini=mock_gemini)

    assert result["search_query"] == "better search query"
    assert result["retry_count"] == 1
    mock_gemini.refine_query.assert_called_once_with("original question", "old query")
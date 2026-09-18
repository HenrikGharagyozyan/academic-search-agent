from app.agents.nodes import verify_evidence_node
from app.schemas.answer import Claim
from app.schemas.document import Chunk


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
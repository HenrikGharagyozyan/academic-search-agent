"""An answer with an internal id in it never reaches the reader.

The stripping in text/cleanup is the first line; this is the second. If an id
survives it, the generation is retried, and if the retry leaks too the answer
is withheld rather than shipped with the id in it.
"""

import logging
from unittest.mock import MagicMock

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk

ID = "72cacfe-cb0a-435f-a3f4-7b7a3bd1bfc0"


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc", text="passage",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


def response(claim_text: str) -> ClaimsResponse:
    return ClaimsResponse(
        summary="s",
        claims=[Claim(text=claim_text, evidence_ids=["c1"], confidence="high")],
        conclusion="k",
    )


STATE = {"question": "q?", "selected_chunks": [chunk("c1")]}


def test_the_reported_leak_is_stripped_before_the_answer_leaves():
    llm = MagicMock()
    llm.generate_answer.return_value = response(
        f"Logical errors are suppressed, allowing fidelity to improve exponentially.[evidence_id: {ID}]"
    )

    out = generate_claims_node(STATE, llm=llm)

    assert out["claims"][0].text == (
        "Logical errors are suppressed, allowing fidelity to improve exponentially."
    )
    assert llm.generate_answer.call_count == 1


def test_a_leak_the_stripping_misses_is_logged_and_regenerated(monkeypatch, caplog):
    # Stand in for a leak form the patterns have never seen.
    monkeypatch.setattr(
        "app.application.agents.nodes.generation.strip_evidence_ids", lambda text: text
    )
    llm = MagicMock()
    llm.generate_answer.side_effect = [
        response(f"Rates fall evidence_id {ID}."),
        response("Rates fall."),
    ]

    with caplog.at_level(logging.WARNING):
        out = generate_claims_node(STATE, llm=llm)

    assert llm.generate_answer.call_count == 2
    assert out["claims"][0].text == "Rates fall."
    assert "survived cleanup" in caplog.text


def test_an_answer_that_keeps_leaking_is_withheld(monkeypatch):
    monkeypatch.setattr(
        "app.application.agents.nodes.generation.strip_evidence_ids", lambda text: text
    )
    llm = MagicMock()
    llm.generate_answer.return_value = response(f"Rates fall evidence_id {ID}.")

    out = generate_claims_node(STATE, llm=llm)

    assert out["claims"] == []
    assert out["summary"] == "" and out["conclusion"] == ""
    assert "internal ids" in out["activity"][-1].label

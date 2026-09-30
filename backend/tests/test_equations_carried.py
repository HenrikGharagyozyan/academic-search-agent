"""A formula the passages state reaches the answer, or its absence is reported.

In an answer on entropy S = k_B ln Ω was not badly formatted — it was not
there. Most of that was selection: the passages stating it never reached the
model. This covers the part that is the model's: given a passage that sets an
equation, an answer without one is regenerated, and if it still has none the
activity trail says so. It must not go missing silently.
"""

import logging
from unittest.mock import MagicMock

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.infrastructure.llm.prompts import ANSWER_PROMPT, SYSTEM_PROMPT

FLAT = " ".join(SYSTEM_PROMPT.split())

WITH_FORMULA = (
    "The entropy is proportional to the natural logarithm of this number:\n"
    "$$S=k_{\\text{B}}\\ln \\Omega$$\n"
    "The proportionality constant is the Boltzmann constant."
)
IN_WORDS = "Entropy is proportional to the logarithm of the number of microstates."
IN_LATEX = "Boltzmann's formula is $$S = k_B \\ln \\Omega$$ where $\\Omega$ counts microstates."


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c1", document_id="doc", text=text,
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


def _response(text: str) -> ClaimsResponse:
    return ClaimsResponse(
        summary="s",
        claims=[Claim(text=text, evidence_ids=["c1"], confidence="high")],
        conclusion="k",
    )


def _state(passage: str) -> dict:
    return {"question": "What is entropy?", "selected_chunks": [_chunk(passage)]}


def test_a_stated_formula_appears_in_the_answer():
    llm = MagicMock()
    llm.generate_answer.return_value = _response(IN_LATEX)

    out = generate_claims_node(_state(WITH_FORMULA), llm=llm)

    assert "$$S = k_B \\ln \\Omega$$" in out["claims"][0].text
    assert llm.generate_answer.call_count == 1
    assert "gives none" not in out["activity"][-1].detail


def test_an_answer_that_drops_it_is_regenerated(caplog):
    llm = MagicMock()
    llm.generate_answer.side_effect = [_response(IN_WORDS), _response(IN_LATEX)]

    with caplog.at_level(logging.WARNING):
        out = generate_claims_node(_state(WITH_FORMULA), llm=llm)

    assert llm.generate_answer.call_count == 2
    assert "$$S = k_B \\ln \\Omega$$" in out["claims"][0].text
    assert "the answer states none" in caplog.text


def test_a_formula_that_stays_missing_does_not_go_missing_silently():
    llm = MagicMock()
    llm.generate_answer.return_value = _response(IN_WORDS)

    out = generate_claims_node(_state(WITH_FORMULA), llm=llm)

    assert out["claims"], "the answer is still worth having"
    assert "the passages state 1 equation and the answer gives none" in out["activity"][-1].detail


def test_passages_without_equations_ask_for_none():
    llm = MagicMock()
    llm.generate_answer.return_value = _response(IN_WORDS)

    out = generate_claims_node(_state("Clausius coined the word entropy in 1865."), llm=llm)

    assert llm.generate_answer.call_count == 1
    assert "gives none" not in out["activity"][-1].detail


def test_the_prompt_makes_a_stated_equation_a_fact_like_any_other():
    assert "that equation MUST appear in the answer" in FLAT
    assert "familiarity is not a reason to leave a stated formula out" in FLAT
    assert "the same as omitting any other fact the evidence provides" in FLAT


def test_the_requirement_is_repeated_after_the_evidence():
    human = ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "PASSAGES"}).to_messages()[-1].content

    assert "must appear in your answer" in " ".join(human[human.index("PASSAGES") :].split())

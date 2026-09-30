"""The answer takes the shape of the question.

"What is energy in physics? what about formula Einstein" came back as five
themed sections, two of which — "Einstein's mass-energy equivalence" and
"Mass-energy equivalence principle" — were one fact told twice. Every rule that
should have prevented that was already in the prompt. What was missing was the
step before them: telling a direct question from a survey.
"""

from unittest.mock import MagicMock

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.infrastructure.llm.prompts import SYSTEM_PROMPT

FLAT = " ".join(SYSTEM_PROMPT.split())


def test_the_prompt_decides_the_shape_before_it_groups():
    assert "FIRST DECIDE THE SHAPE OF THE ANSWER" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("FIRST DECIDE THE SHAPE") < SYSTEM_PROMPT.index("GROUP BY DIRECTION")


def test_a_direct_question_gets_one_claim_with_no_theme():
    assert "ONE claim with no theme" in FLAT
    assert "one thing described N times" in FLAT


def test_a_survey_question_still_gets_themed_sections():
    # The fix must teach when to use sections, not remove them.
    assert "deserves multiple themed sections" in FLAT
    assert "ONE SECTION PER ASPECT THE EVIDENCE COVERS" in SYSTEM_PROMPT
    assert "not a reason to turn a direct question into a survey" in FLAT


def test_one_fact_under_two_theme_wordings_is_named_as_a_duplicate():
    assert "Einstein's mass-energy equivalence" in FLAT
    assert "Mass-energy equivalence principle" in FLAT
    assert "one claim that should have been written once" in FLAT


def _generate(shape: str, themes: list[str]):
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        answer_shape=shape,
        summary="s",
        claims=[Claim(text="t", evidence_ids=["c1"], confidence="high", theme=t) for t in themes],
        conclusion="k",
    )
    chunk = Chunk(
        chunk_id="c1", document_id="doc", text="p",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )
    return generate_claims_node({"question": "q?", "selected_chunks": [chunk]}, llm=llm)


def test_the_shape_is_the_first_thing_the_model_writes():
    # Structured output is generated in schema order, so first here means the
    # model decides before it writes a claim.
    assert list(ClaimsResponse.model_json_schema()["properties"])[0] == "answer_shape"


def test_a_direct_answer_has_no_sections_even_if_the_model_wrote_themes():
    out = _generate("direct", ["Definition of energy", "Einstein's mass-energy equivalence"])

    assert [c.theme for c in out["claims"]] == ["", ""]


def test_a_survey_answer_keeps_its_sections():
    out = _generate("survey", ["real-time decoding", "Physical qubit overhead"])

    assert [c.theme for c in out["claims"]] == ["Real-time decoding", "Physical qubit overhead"]


def test_the_activity_trail_says_which_shape_was_chosen():
    out = _generate("direct", [""])

    assert out["activity"][-1].detail == "answered as a direct question"


def test_the_prompt_ties_the_field_to_the_rule():
    assert '`answer_shape` is "direct"' in FLAT
    assert "one claim per thing asked, never one per source" in FLAT

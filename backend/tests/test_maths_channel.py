"""LaTeX survives the JSON it travels in.

The raw strings below are what a live model returned for "what is entropy in
statistical mechanics" — copied from the output before any post-processing. It
wrote its commands with backslashes, JSON decoded them as escapes, and the
filters then stripped the control characters: the reader got
"$S = k_B ext{ln} ext{Ω}$" and "$f$" where Ω had been.
"""

import logging
from unittest.mock import MagicMock

import pytest

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.text.equations import (
    display_equations,
    mentions_in_plain_text,
    plain_text_equations,
    skeleton,
    states_an_equation,
    typesets,
)
from app.domain.text.latex import has_damaged_maths, restore_latex, to_channel_notation
from app.infrastructure.llm.base import LangChainLLMProvider
from app.infrastructure.llm.prompts import SYSTEM_PROMPT

# Verbatim from the model: "\t" here is a real tab, "\x15" a real control byte.
EATEN_TEXT = "$S = k_B \text{ln} \\, \text{Ω}$"
LOST_OMEGA = "where $\x15f$ is the number of microstates"
LOST_BACKSLASH = "$\x1aDelta S$"


# --- repairing what JSON ate -------------------------------------------------


def test_a_command_json_decoded_as_a_tab_is_put_back():
    assert restore_latex(EATEN_TEXT) == "$S = k_B \\text{ln} \\, \\text{Ω}$"


@pytest.mark.parametrize(
    ("eaten", "command"),
    [
        ("$\x0crac{1}{2}$", "$\\frac{1}{2}$"),
        ("$a \x08eta$", "$a \\beta$"),
        ("$\rho$", "$\\rho$"),
        ("$a \neq b$", "$a \\neq b$"),
        ("$\nabla f$", "$\\nabla f$"),
        ("$2 \times 3$", "$2 \\times 3$"),
    ],
)
def test_every_json_escape_that_starts_a_command_is_reversed(eaten, command):
    assert restore_latex(eaten) == command


def test_a_command_that_lost_only_its_backslash_is_put_back():
    assert restore_latex(LOST_BACKSLASH) == "$\\Delta S$"
    assert not has_damaged_maths(LOST_BACKSLASH)


def test_a_newline_in_display_maths_is_left_as_a_newline():
    text = "$$\na = b\n$$"
    assert restore_latex(text) == text


def test_a_formula_that_lost_a_symbol_outright_is_reported_as_damaged():
    assert has_damaged_maths(LOST_OMEGA)
    assert not has_damaged_maths(EATEN_TEXT)
    assert not has_damaged_maths("Plain prose with $S = k_B \\ln \\Omega$.")


# --- the notation the model reads is the notation it writes ------------------


def test_passages_are_shown_to_the_model_in_at_notation():
    assert to_channel_notation("$S=k_{\\text{B}}\\ln \\Omega$") == "$S=k_{@text{B}}@ln @Omega$"


def test_the_doubled_backslashes_of_scraped_markdown_are_one_command():
    assert to_channel_notation("$\\\\frac{1}{2}$") == "$@frac{1}{2}$"


def test_a_fenced_block_in_a_passage_is_not_rewritten():
    text = "```text\nC:\\Users\\x\n```"
    assert to_channel_notation(text) == text


def test_the_evidence_block_carries_no_backslash_commands():
    llm = MagicMock()
    provider = LangChainLLMProvider(llm, provider_name="p", model_name="m")
    chunk = Chunk(
        chunk_id="c1", document_id="d", text="Boltzmann: $$S=k_{\\text{B}}\\ln \\Omega$$",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )

    provider.generate_answer("q?", [chunk])

    prompt = llm.with_structured_output.return_value.invoke.call_args.args[0]
    human = prompt.to_messages()[-1].content
    assert "$$S=k_{@text{B}}@ln @Omega$$" in human
    assert "\\ln" not in human
    # The passage itself is untouched: the reader's citation shows the original.
    assert "\\ln" in chunk.text


def test_the_at_rule_opens_the_maths_block():
    block = SYSTEM_PROMPT[SYSTEM_PROMPT.index("MATHEMATICAL NOTATION") :]
    rule = block.index("WRITE EVERY LATEX COMMAND WITH @")

    assert rule < block.index("A defining or governing equation")
    assert rule < block.index("CARRY THE EVIDENCE'S MATHS ACROSS")
    assert "JSON string" in block


# --- recognising an equation however it is written ---------------------------


@pytest.mark.parametrize(
    "written",
    ["S = k_B ln Ω", "S=k_{\\text{B}}\\ln \\Omega", "S = k_B @ln @Omega", "S=kBln\u2061Ω"],
)
def test_one_equation_has_one_skeleton(written):
    assert skeleton(written) == "S=kBlnΩ"


def test_typeset_and_plain_are_told_apart():
    answer = "The law is $$F = ma$$. Boltzmann wrote S = k_B ln Ω."

    assert typesets(answer, "F = ma")
    assert not typesets(answer, "S = k_B ln Ω")
    assert mentions_in_plain_text(answer, "S = k_B ln Ω")
    assert not mentions_in_plain_text(answer, "F = ma")


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("is expressed as E = mc², illustrating", ["E = mc²"]),
        ("usually written F=ma, where F is force", ["F=ma"]),
        ("so that ΔS ≥ 0 for an isolated system", ["ΔS ≥ 0"]),
        ("rearranged, a = F/m gives", ["a = F/m"]),
    ],
)
def test_an_equation_left_as_ordinary_characters_is_found(text, found):
    assert plain_text_equations(text) == found


@pytest.mark.parametrize(
    "text",
    [
        "The law is $$F = ma$$ where $m$ is mass.",
        "See https://example.org/?id=abc&y=z for details.",
        "Adapters of rank r = 8 were trained on n = 120 samples.",
        "```text\nx = y\n```",
        "A key=value store.",
    ],
)
def test_prose_settings_urls_and_set_maths_are_not_mistaken_for_one(text):
    assert plain_text_equations(text) == []


def test_display_equations_in_a_passage_are_listed():
    passage = "Of this number:\n$$S=k_{\\text{B}}\\ln \\Omega$$\nwhere $k$ is a constant."

    assert display_equations(passage) == ["S=k_{\\text{B}}\\ln \\Omega"]
    assert states_an_equation(passage)
    assert not states_an_equation("where $k$ is a constant")


# --- the answer is regenerated when a formula is wrong -----------------------


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="c1", document_id="doc", text="passage",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


def _response(text: str) -> ClaimsResponse:
    return ClaimsResponse(
        summary="s",
        claims=[Claim(text=text, evidence_ids=["c1"], confidence="high")],
        conclusion="k",
    )


STATE = {"question": "q?", "selected_chunks": [_chunk()]}


def test_a_repairable_formula_costs_no_second_generation():
    llm = MagicMock()
    llm.generate_answer.return_value = _response(f"Boltzmann: {EATEN_TEXT}")

    out = generate_claims_node(STATE, llm=llm)

    assert llm.generate_answer.call_count == 1
    assert out["claims"][0].text == "Boltzmann: $S = k_B \\text{ln} \\, \\text{Ω}$"


def test_a_formula_damaged_beyond_repair_is_regenerated(caplog):
    llm = MagicMock()
    llm.generate_answer.side_effect = [
        _response(LOST_OMEGA),
        _response("where $\\Omega$ is the number of microstates"),
    ]

    with caplog.at_level(logging.WARNING):
        out = generate_claims_node(STATE, llm=llm)

    assert llm.generate_answer.call_count == 2
    assert out["claims"][0].text == "where $\\Omega$ is the number of microstates"
    assert "damaged beyond repair" in caplog.text


def test_an_equation_left_as_plain_text_is_regenerated(caplog):
    llm = MagicMock()
    llm.generate_answer.side_effect = [
        _response("The formula is E = mc², where c is the speed of light."),
        _response("The formula is $$E = mc^2$$ where $c$ is the speed of light."),
    ]

    with caplog.at_level(logging.WARNING):
        out = generate_claims_node(STATE, llm=llm)

    assert llm.generate_answer.call_count == 2
    assert "$$E = mc^2$$" in out["claims"][0].text
    assert "left as plain text" in caplog.text


def test_the_reader_is_told_when_the_damage_survives_the_retry():
    llm = MagicMock()
    llm.generate_answer.return_value = _response(LOST_OMEGA)

    out = generate_claims_node(STATE, llm=llm)

    assert out["claims"], "one lost symbol is not worth the whole answer"
    assert "may be missing a symbol" in out["activity"][-1].detail


def test_the_better_attempt_is_kept_when_the_retry_is_worse():
    llm = MagicMock()
    llm.generate_answer.side_effect = [
        _response("The formula is E = mc², where c is the speed of light."),
        _response("As shown in evidence_id 3, the rate falls."),
    ]

    out = generate_claims_node(STATE, llm=llm)

    assert out["claims"][0].text.startswith("The formula is E = mc²")


def test_a_failed_retry_does_not_cost_the_answer_in_hand():
    llm = MagicMock()
    llm.generate_answer.side_effect = [
        _response("The formula is E = mc², where c is the speed of light."),
        RuntimeError("503"),
    ]

    out = generate_claims_node(STATE, llm=llm)

    assert out["claims"][0].text.startswith("The formula is E = mc²")


# --- the rule is repeated where it is acted on -------------------------------


def test_the_maths_rules_are_the_last_thing_the_model_reads():
    from app.infrastructure.llm.prompts import ANSWER_PROMPT

    human = ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "PASSAGES"}).to_messages()[-1].content

    after_evidence = human[human.index("PASSAGES") :]
    assert "$$E = mc^2$$" in after_evidence
    assert "$$S = k_B @ln @Omega$$" in after_evidence
    assert "never a backslash" in after_evidence


@pytest.mark.parametrize(
    ("eaten", "command"),
    [
        ("$\x08aOmega$", "$\\Omega$"),
        ("$\x1aDelta S$", "$\\Delta S$"),
        ("$\tilde{x}$", "$\\tilde{x}$"),
    ],
)
def test_a_name_that_survived_behind_the_escape_is_recognised(eaten, command):
    assert restore_latex(eaten) == command


def test_an_escape_with_nothing_recognisable_behind_it_is_damage():
    assert has_damaged_maths("the number of microstates ($\x0c$)")
    assert has_damaged_maths("$$\x0c S \x0c 0$$")


def test_a_command_the_list_does_not_know_is_still_put_back():
    # From a live answer on entropy: both attempts arrived like this, and the
    # reader would have seen "riangle S".
    assert restore_latex("$$\triangle S = @frac{q_{rev}}{T}$$") == (
        "$$\\triangle S = \\frac{q_{rev}}{T}$$"
    )
    assert restore_latex("$@Delta S \textgreater 0$") == "$\\Delta S \\textgreater 0$"
    assert not has_damaged_maths("$$\triangle S_{total} @geq 0$$")


def test_a_newline_before_an_unknown_word_is_not_turned_into_a_command():
    text = "$$\nx = y\n$$"
    assert restore_latex(text) == text

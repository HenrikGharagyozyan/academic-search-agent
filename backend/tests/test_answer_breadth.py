"""The answer is meant to survey an area, not recite one paper.

The reported case produced four claims about one method with no sense of what
else exists. Grounding is not the thing to relax for that — it is the product —
so the change is in what the model is asked to produce: claims tagged with the
direction they belong to, and a conclusion that is allowed to synthesise.
"""

from unittest.mock import MagicMock

from app.application.agents.nodes import generate_claims_node, verify_evidence_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.infrastructure.llm.prompts import SYSTEM_PROMPT


def chunk(chunk_id: str, text: str = "passage") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc", text=text,
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


# --- the theme survives the pipeline ---------------------------------------


def test_a_claims_theme_reaches_the_answer():
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        summary="s",
        claims=[
            Claim(text="a", evidence_ids=["c1"], confidence="high", theme="online learning"),
            Claim(text="b", evidence_ids=["c1"], confidence="high", theme="online learning"),
            Claim(text="c", evidence_ids=["c1"], confidence="medium", theme="dynamic matrices"),
        ],
        conclusion="k",
    )

    out = generate_claims_node({"question": "q?", "selected_chunks": [chunk("c1")]}, llm=llm)

    assert [c.theme for c in out["claims"]] == [
        "online learning", "online learning", "dynamic matrices",
    ]


def test_verification_does_not_strip_the_theme():
    """verify_evidence rebuilds each claim to trim its citations; the theme has
    to come through that copy or the grouping disappears from the answer."""
    state = {
        "selected_chunks": [chunk("real")],
        "claims": [
            Claim(
                text="grounded", evidence_ids=["real", "invented"],
                confidence="high", theme="online learning",
            )
        ],
    }

    out = verify_evidence_node(state)

    assert out["claims"][0].theme == "online learning"
    assert out["claims"][0].evidence_ids == ["real"]


def test_a_theme_is_optional():
    # A single-subject topic should not be forced into headings.
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        summary="s",
        claims=[Claim(text="a", evidence_ids=["c1"], confidence="high")],
        conclusion="k",
    )

    out = generate_claims_node({"question": "q?", "selected_chunks": [chunk("c1")]}, llm=llm)

    assert out["claims"][0].theme == ""


# --- what the prompt actually asks for -------------------------------------


def test_the_prompt_asks_for_claims_to_be_grouped_by_direction():
    assert "GROUP BY DIRECTION" in SYSTEM_PROMPT
    assert "theme" in SYSTEM_PROMPT


def test_the_prompt_forbids_flattening_distinct_work_into_false_consensus():
    assert "consensus" in SYSTEM_PROMPT


def test_the_prompt_lets_the_conclusion_synthesise_but_not_invent():
    assert "where it is heading" in SYSTEM_PROMPT
    # The licence to synthesise has to arrive with its limit attached.
    assert "invention" in SYSTEM_PROMPT


def test_the_prompt_asks_for_thin_evidence_to_be_reported_as_such():
    """The reported failure was a confident answer built on one paper. Saying
    "the sources all cover one method" is the more useful outcome."""
    assert "thin or one-sided" in SYSTEM_PROMPT


def test_the_prompt_still_requires_every_claim_to_cite_evidence():
    # The point of the change is breadth, not loosening grounding.
    assert "Every claim MUST cite at least one evidence_id" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT


def test_the_prompt_asks_for_equations_in_the_evidence_to_be_reproduced():
    assert "CARRY THE EVIDENCE'S MATHS ACROSS" in SYSTEM_PROMPT


# --- maths reaches the model intact ----------------------------------------


def test_equations_in_a_scraped_page_survive_into_the_evidence():
    """Nothing between the scrape and the prompt touches the passage text, and
    this pins that: the LaTeX helpers run on the model's output, not its input."""
    from app.domain.text.chunker import chunk_lines
    from app.domain.text.splitter import split_into_lines

    page = (
        "The transmission matrix $T$ relates the fields:\n\n"
        "$$\nE_{out} = T \\, E_{in}\n$$\n\n"
        "with $T \\in \\mathbb{C}^{N \\times M}$ and enhancement "
        "$\\eta = \\frac{\\pi}{4} N$.\n"
    )

    chunks = chunk_lines("doc", split_into_lines(page), "https://x", "T")
    body = "\n".join(c.text for c in chunks)

    for fragment in ("E_{out} = T", r"\mathbb{C}", r"\frac{\pi}{4}", "$$"):
        assert fragment in body, f"{fragment} was lost before generation"


# --- process diagrams ------------------------------------------------------


def test_the_prompt_offers_a_diagram_for_a_real_sequence():
    assert "PROCESS DIAGRAMS" in SYSTEM_PROMPT
    assert "@rightarrow" in SYSTEM_PROMPT


def test_the_prompt_restricts_diagrams_to_actual_processes():
    """Drawing a set of alternatives as a chain would assert an order the
    sources never claimed."""
    assert "not a process" in SYSTEM_PROMPT
    assert "When in doubt, prose" in SYSTEM_PROMPT


def test_the_diagram_example_survives_prompt_templating():
    """The braces in \\boxed{...} have to be escaped for ChatPromptTemplate, and
    getting that wrong turns the example into a KeyError or a literal {{}}."""
    from app.infrastructure.llm.prompts import ANSWER_PROMPT

    rendered = str(ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "e"}))

    assert "@boxed{measurement}" in rendered
    assert "{{" not in rendered.split("PROCESS DIAGRAMS")[1][:600]


def test_a_boxed_chain_from_the_model_becomes_renderable_latex():
    """End to end through the output filters: what the model writes with @ has to
    come out as backslash commands KaTeX can render."""
    from app.domain.text.cleanup import strip_evidence_ids
    from app.domain.text.latex import restore_latex

    written = (
        "The loop is $$@boxed{probe} @rightarrow @boxed{estimate T} @rightarrow "
        "@boxed{apply correction}$$ and it repeats."
    )

    out = strip_evidence_ids(restore_latex(written))

    assert r"\boxed{probe}" in out
    assert r"\rightarrow" in out
    assert "@" not in out


# --- depth per claim -------------------------------------------------------


def test_the_prompt_asks_for_a_paragraph_per_approach():
    """Claims were arriving as one-sentence restatements of a paper's title."""
    assert "WRITE A PARAGRAPH, NOT A HEADLINE" in SYSTEM_PROMPT
    for part in ("WHAT it is", "WHY IT MATTERS", "HOW IT DIFFERS"):
        assert part in SYSTEM_PROMPT


def test_asking_for_depth_arrives_with_its_limit():
    """Asking for longer claims invites padding a thin passage into a confident
    paragraph, which is worse than a short honest one."""
    assert "never from padding" in SYSTEM_PROMPT
    assert "set confidence accordingly" in SYSTEM_PROMPT
    assert "Never invent a number" in SYSTEM_PROMPT


# --- the trajectory --------------------------------------------------------


def test_the_prompt_asks_for_the_direction_of_travel():
    assert "NAME THE TRAJECTORY" in SYSTEM_PROMPT
    assert "progression" in SYSTEM_PROMPT


def test_the_trajectory_may_not_be_manufactured():
    """Methods that are merely different are not a sequence, and drawing them as
    one asserts a history the sources never claimed."""
    assert "Do not manufacture a progression" in SYSTEM_PROMPT
    assert "parallel approaches are parallel" in SYSTEM_PROMPT


def test_the_trajectory_example_survives_templating():
    from app.infrastructure.llm.prompts import ANSWER_PROMPT

    rendered = str(ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "e"}))

    assert "@boxed{static measurement}" in rendered
    assert "@boxed{online adaptation}" in rendered


# --- one section per aspect, not a tidy small number -----------------------


def test_the_prompt_no_longer_caps_the_number_of_sections():
    """It used to say "keep the themes few: three to five for a broad topic",
    and the model obeyed: 24 on-topic passages became 3 claims."""
    assert "keep the themes few" not in SYSTEM_PROMPT
    assert "ONE SECTION PER ASPECT" in SYSTEM_PROMPT
    assert "Eight or ten" in SYSTEM_PROMPT


def test_the_prompt_says_when_to_merge_and_when_not_to():
    """Lifting the cap must not turn into one claim per passage."""
    assert "restating the same fact are one claim" in SYSTEM_PROMPT
    assert "different problems are two claims" in SYSTEM_PROMPT


# --- markdown structure ----------------------------------------------------


def test_the_prompt_declares_a_claim_to_be_markdown():
    assert "A CLAIM IS MARKDOWN" in SYSTEM_PROMPT


def test_the_prompt_offers_lists_tables_and_text_diagrams():
    for structure in ("BULLET LIST", "MARKDOWN TABLE", "FENCED ```text BLOCK"):
        assert structure in SYSTEM_PROMPT


def test_diagrams_are_told_to_use_spaces_because_tabs_do_not_survive():
    """restore_latex strips control characters from model output, tab included —
    a tab-aligned diagram would arrive collapsed."""
    from app.domain.text.latex import restore_latex

    assert "Use spaces, never tabs" in SYSTEM_PROMPT
    assert restore_latex("a\tb") == "ab", "if tabs now survive, the warning can go"


def test_structure_is_offered_with_a_reason_not_as_a_default():
    assert "Structure earns its place" in SYSTEM_PROMPT
    assert "PROSE for everything else" in SYSTEM_PROMPT


def test_both_diagram_forms_have_a_stated_niche():
    """A text block for architecture, a LaTeX chain for quantities flowing."""
    assert "A ```text block for anything with branches" in SYSTEM_PROMPT
    assert "reads as \\\nmathematics" in SYSTEM_PROMPT or "reads as" in SYSTEM_PROMPT


def test_the_model_is_told_not_to_narrate_missing_numbers():
    """A live answer contained "although no specific data is provided", which is
    the anti-padding rule firing awkwardly."""
    assert "do not write that a number was not provided" in SYSTEM_PROMPT


def test_a_fenced_diagram_survives_the_output_filters():
    """The filters run on model output, so a diagram has to come through them
    unchanged — no @-substitution, no stripped characters."""
    from app.domain.text.cleanup import strip_evidence_ids
    from app.domain.text.latex import restore_latex

    written = (
        "The loop:\n\n```text\nPhysical qubits\n      |\n      v\n"
        "Syndrome measurement --> Decoder --> Correction\n```\n\nand it repeats."
    )

    out = strip_evidence_ids(restore_latex(written))

    assert "```text" in out
    assert "Syndrome measurement --> Decoder --> Correction" in out
    assert "      |" in out, "indentation must survive"


def test_a_markdown_table_survives_the_output_filters():
    from app.domain.text.cleanup import strip_evidence_ids
    from app.domain.text.latex import restore_latex

    written = (
        "| Goal | Trade-off |\n|---|---|\n"
        "| Higher threshold | More complex hardware |\n"
        "| Fewer qubits | Harder decoding |"
    )

    out = strip_evidence_ids(restore_latex(written))

    assert out.count("|") == written.count("|")
    assert "Higher threshold" in out

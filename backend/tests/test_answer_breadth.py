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

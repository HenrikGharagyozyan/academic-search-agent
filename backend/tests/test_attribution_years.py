"""An attribution's year is its source's, not the model's.

The same question, "latest research on transmission matrix engineering", cited
Cheng et al. as 2022 in one run and 2023 in the next, and Resisi et al. as 2019
and 2020. The prompt now shows each source's year; this holds the answer to it.
"""

from unittest.mock import MagicMock

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.attribution import correct_years
from app.domain.citation import SourceCitation
from app.domain.documents import Chunk

CHENG = SourceCitation(authors=["Cheng, Zhe", "Li, Wen", "Wu, Hao"], year=2023)
RESISI_PREPRINT = SourceCitation(authors=["Resisi, Shachar", "Bromberg, Yaron"], year=2019)
RESISI_PAPER = SourceCitation(authors=["Shachar Resisi", "Yaron Bromberg"], year=2020)


def test_a_wrong_year_is_set_to_the_sources_own():
    text, changed = correct_years("Cheng et al. (2022) learn the matrix online.", [CHENG])

    assert text == "Cheng et al. (2023) learn the matrix online."
    assert changed == 1


def test_the_parenthetical_form_is_corrected_too():
    text, _ = correct_years("The matrix is learned online (Cheng et al., 2022).", [CHENG])

    assert text == "The matrix is learned online (Cheng et al., 2023)."


def test_a_right_year_is_left_alone():
    assert correct_years("Cheng et al. (2023) show it.", [CHENG]) == ("Cheng et al. (2023) show it.", 0)


def test_an_author_of_two_cited_works_keeps_either_year():
    both = [RESISI_PREPRINT, RESISI_PAPER]

    assert correct_years("Resisi et al. (2019) ...", both)[1] == 0
    assert correct_years("Resisi et al. (2020) ...", both)[1] == 0
    # Neither year: ambiguous which was meant, so it is not guessed.
    assert correct_years("Resisi et al. (2021) ...", both)[1] == 0


def test_a_name_no_cited_source_declares_is_left_alone():
    """A passage may cite other work by name; only its own authors are checked."""
    assert correct_years("Popoff et al. (2010) measured it first.", [CHENG])[1] == 0


def test_the_generated_answer_carries_the_sources_year():
    chunk = Chunk(
        chunk_id="ev_1", document_id="d", text="Online learning of the transmission matrix.",
        start_line=1, end_line=2, source_url="https://opg.optica.org/x", title="Online learning",
        citation=CHENG,
    )
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        answer_shape="survey",
        summary="Cheng et al. (2022) lead the online approach.",
        claims=[Claim(
            text="Cheng et al. (2022) learn the matrix online.",
            evidence_ids=["ev_1"], confidence="high", theme="Online learning",
        )],
        conclusion="Online learning (Cheng et al., 2022) is the main direction.",
    )

    out = generate_claims_node({"question": "q?", "selected_chunks": [chunk]}, llm=llm)

    assert out["claims"][0].text == "Cheng et al. (2023) learn the matrix online."
    assert out["summary"] == "Cheng et al. (2023) lead the online approach."
    assert out["conclusion"] == "Online learning (Cheng et al., 2023) is the main direction."


# --- the heading and the body give one attribution --------------------------


def test_an_attribution_in_the_heading_moves_into_the_claim():
    """A survey answer headed a section "Jiang et al. (2026)" while its claim
    named nobody. The heading is a direction shared by the section's claims;
    the claim is where its own source is named."""
    from app.domain.attribution import move_attribution_out_of_theme

    theme, text = move_attribution_out_of_theme(
        "Adaptive correction — Jiang et al. (2026)",
        "A method restores the focus after the fiber bends.",
    )

    assert theme == "Adaptive correction"
    assert text == "A method restores the focus after the fiber bends (Jiang et al., 2026)."


def test_an_attribution_the_claim_already_gives_is_only_dropped_from_the_heading():
    from app.domain.attribution import move_attribution_out_of_theme

    theme, text = move_attribution_out_of_theme(
        "Jiang et al. (2026): adaptive correction",
        "Jiang et al. (2026) restore the focus after the fiber bends.",
    )

    assert theme == "adaptive correction"
    assert text == "Jiang et al. (2026) restore the focus after the fiber bends."


def test_the_moved_attribution_takes_the_sources_year():
    chunk = Chunk(
        chunk_id="ev_1", document_id="d", text="Restoring the focus.", start_line=1, end_line=1,
        source_url="https://x.org", title="T",
        citation=SourceCitation(authors=["Jiang, Wei", "Zhou, Li", "Ma, Yu"], year=2025),
    )
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        answer_shape="survey", summary="s", conclusion="c",
        claims=[Claim(
            text="A method restores the focus after the fiber bends.", evidence_ids=["ev_1"],
            confidence="medium", theme="Jiang et al. (2026)",
        )],
    )

    claim = generate_claims_node({"question": "q?", "selected_chunks": [chunk]}, llm=llm)["claims"][0]

    assert claim.text == "A method restores the focus after the fiber bends (Jiang et al., 2025)."
    assert "Jiang" not in claim.theme


def test_a_heading_that_merely_mentions_a_year_is_not_an_attribution():
    from app.domain.attribution import move_attribution_out_of_theme

    for heading in ("QLoRA 2023 variants", "Benchmarks since 2020", "GPT 2024 models"):
        assert move_attribution_out_of_theme(heading, "Text.") == (heading, "Text.")

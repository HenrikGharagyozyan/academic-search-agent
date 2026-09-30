"""A theme becomes a numbered section heading: "### 1. Error rates and sources".

The reported answer showed "1.error rates and sources" — lowercase, because the
prompt's own examples were, and the number is added by the interface, so the
model has to hand over the heading's words and nothing else.
"""

from unittest.mock import MagicMock

import pytest

from app.application.agents.nodes import generate_claims_node
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.text.headings import tidy_theme
from app.infrastructure.llm.prompts import SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("written", "heading"),
    [
        ("error rates and sources", "Error rates and sources"),
        ("Error rates and sources", "Error rates and sources"),
        ("1.error rates and sources", "Error rates and sources"),
        ("1. Error rates and sources", "Error rates and sources"),
        ("### 2. real-time decoding", "Real-time decoding"),
        ("3) magic-state distillation.", "Magic-state distillation"),
        ("- surface codes:", "Surface codes"),
        ("  logical error suppression  ", "Logical error suppression"),
    ],
)
def test_a_theme_is_tidied_into_heading_words(written, heading):
    assert tidy_theme(written) == heading


@pytest.mark.parametrize("theme", ["qLDPC codes", "mRNA vaccines", "LoRA adapters", "NeRF in 2025"])
def test_deliberate_casing_is_kept(theme):
    assert tidy_theme(theme) == theme


def test_a_year_leading_the_theme_is_not_taken_for_numbering():
    assert tidy_theme("2025 benchmarks") == "2025 benchmarks"


def test_an_empty_theme_stays_empty():
    assert tidy_theme("") == ""
    assert tidy_theme("### ") == ""


def test_the_generated_answer_carries_heading_ready_themes():
    llm = MagicMock()
    llm.generate_answer.return_value = ClaimsResponse(
        summary="s",
        claims=[
            Claim(text="a", evidence_ids=["c1"], confidence="high", theme="1.error rates and sources"),
            Claim(text="b", evidence_ids=["c1"], confidence="high", theme="error rates and sources"),
            Claim(text="c", evidence_ids=["c1"], confidence="high", theme=""),
        ],
        conclusion="k",
    )
    chunk = Chunk(
        chunk_id="c1", document_id="doc", text="p",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )

    out = generate_claims_node({"question": "q?", "selected_chunks": [chunk]}, llm=llm)

    # Both spellings land on one heading, so the two claims stay one section.
    assert [c.theme for c in out["claims"]] == [
        "Error rates and sources", "Error rates and sources", "",
    ]


def test_the_prompt_asks_for_heading_words_in_sentence_case():
    assert "### 1. Physical qubit overhead" in " ".join(SYSTEM_PROMPT.split())
    assert "sentence case" in SYSTEM_PROMPT
    assert "No number" in SYSTEM_PROMPT


def test_the_prompts_theme_examples_are_capitalised():
    # The model copies the examples' casing; lowercase ones are how the bug began.
    assert '"physical qubit overhead"' not in SYSTEM_PROMPT
    assert '"Physical qubit overhead"' in SYSTEM_PROMPT

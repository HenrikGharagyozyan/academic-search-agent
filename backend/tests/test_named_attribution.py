"""A source is cited by name when the passage gives one.

The same question had come back once as "Valzania and Gigan (2023)…", "Wang et
al. (2026) introduce…" and once as "as highlighted in source", "as noted in
evidence". No edit had weakened a rule: the prompt had never asked for names.
"""

import re

from app.infrastructure.llm.prompts import ANSWER_PROMPT, SYSTEM_PROMPT

FLAT = " ".join(SYSTEM_PROMPT.split())


def _after_evidence() -> str:
    human = ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "PASSAGES"}).to_messages()[-1].content
    return " ".join(human[human.index("PASSAGES") :].split())


def test_the_prompt_asks_for_the_source_by_name():
    assert "ATTRIBUTE BY NAME" in SYSTEM_PROMPT
    assert "an author, a publication, a journal, a group or a year" in FLAT
    assert "cite them by name" in FLAT


def test_the_vague_substitutes_are_named_as_forbidden():
    for phrase in ('"as shown in source"', '"as noted in evidence"'):
        assert phrase in FLAT


def test_the_rule_sits_with_the_rules_for_claims():
    rule = SYSTEM_PROMPT.index("ATTRIBUTE BY NAME")

    assert SYSTEM_PROMPT.index("Every claim MUST cite") < rule < SYSTEM_PROMPT.index("3. CONCLUSION")


def test_it_does_not_license_an_invented_author():
    # The other half matters as much: a passage that names nobody must not be
    # given a name to satisfy the rule.
    assert "name only what the passage itself names" in FLAT
    assert "Never invent a number, a date, an author" in FLAT


def test_the_rule_is_repeated_after_the_evidence():
    after = _after_evidence()

    assert "name them in your claim" in after
    assert '"as noted in evidence"' in after
    assert "if the passage names nobody" in after


def test_the_maths_reminder_is_still_there():
    assert "$$E = mc^2$$" in _after_evidence()


# A citation shaped like a real one: "Surname (2023)", "Surname et al. (2023)",
# "Surname and Surname (2023)".
_CITATION = re.compile(r"\b[A-Z][a-z]+(?: et al\.| and [A-Z][a-z]+)? \(\d{4}\)")


def test_the_examples_name_nobody_a_model_could_cite():
    """With "Valzania and Gigan (2023)" as the example, an answer on LoRA cited
    Valzania and Gigan for QLoRA. Nothing checks the names beside a citation,
    so the prompt must not offer one."""
    assert not _CITATION.findall(SYSTEM_PROMPT)
    assert not _CITATION.findall(_after_evidence())


def test_the_placeholders_are_declared_as_placeholders():
    assert "<Surname> et al. (<year>)" in SYSTEM_PROMPT
    assert "never text to copy" in FLAT
    assert "must appear in the passage you cite" in FLAT


def test_the_source_line_is_named_as_the_record_of_the_year():
    assert 'carries a "Source:" line' in FLAT
    assert "never another year from the text or from memory" in FLAT
    assert '"Source:" line gives its authors and year exactly' in _after_evidence()


def test_a_theme_is_told_to_name_a_direction_not_a_source():
    assert "A theme names the direction, never a source: no author, no year" in FLAT

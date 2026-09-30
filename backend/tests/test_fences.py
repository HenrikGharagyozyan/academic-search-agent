"""Fenced blocks are verbatim, and the prose filters must run around them.

Found by a test written for the diagram instruction: the whitespace-collapsing
rule that tidies up after an evidence id was removed turned an indented ASCII
diagram into three lines indented by one space.
"""

from app.domain.text.cleanup import strip_evidence_ids
from app.domain.text.fences import outside_fences
from app.domain.text.latex import restore_latex

DIAGRAM = """The pipeline:

```text
Physical qubits
      |
      v
Syndrome measurement --> Decoder --> Correction
```

and it repeats."""


def test_indentation_inside_a_fence_survives():
    out = strip_evidence_ids(restore_latex(DIAGRAM))

    assert "      |" in out
    assert "      v" in out


def test_a_box_drawing_diagram_survives_intact():
    written = (
        "```text\n"
        "┌─────────────────────┐\n"
        "│  logical gates      │\n"
        "├─────────────────────┤\n"
        "│  physical hardware  │\n"
        "└─────────────────────┘\n"
        "```"
    )

    assert strip_evidence_ids(restore_latex(written)) == written


def test_an_at_sign_inside_a_fence_is_not_read_as_a_latex_command():
    written = "```text\nuser@host --> queue\n```"

    assert "@host" in restore_latex(written)


def test_prose_around_a_fence_is_still_filtered():
    b = "005fe97b-b07a-45e4-881b-d6d829c6ab97"
    written = f"Before (evidence_ids: {b}).\n\n```text\n  a  b\n```\n\nAfter $@alpha$."

    out = strip_evidence_ids(restore_latex(written))

    assert "Before." in out, "the id outside the fence must still go"
    assert r"$\alpha$" in out, "maths outside the fence must still be restored"
    assert "  a  b" in out, "the fence itself must be untouched"


def test_text_with_no_fence_is_transformed_whole():
    assert outside_fences("hello", str.upper) == "HELLO"


def test_an_unclosed_fence_protects_the_rest():
    """Which is what a Markdown renderer does with it, and the safer way to be
    wrong: mangling a half-written diagram is worse than leaving it alone."""
    out = outside_fences("before\n```text\n  spaced  ", str.upper)

    assert out.startswith("BEFORE")
    assert "  spaced  " in out


def test_several_fences_in_one_answer():
    out = outside_fences("a\n```\nx\n```\nb\n```\ny\n```\nc", str.upper)

    assert "\nx\n" in out and "\ny\n" in out
    assert "A" in out and "B" in out and "C" in out

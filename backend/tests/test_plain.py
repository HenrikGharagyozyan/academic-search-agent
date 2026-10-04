"""Every input here is a real title or snippet Firecrawl returned for the
question "latest research on transmission matrix engineering". They are the
reason this module exists: rendered as-is, the activity trail was mostly
mailto: links and table pipes.
"""

import pytest

from app.domain.text.plain import to_label

ARXIV_TITLE = (
    "# Title:Wavefront shaping in multimode fibers by transmission matrix engineering "
    "| | | |-|-| | Subjects: | Optics (physics.optics); Quantum Physics (quant-ph) | | "
    "Cite as: | [arXiv:1910.02798](https://arxiv.org/abs/1910.02798) [physics.optics] | "
    "| | [[https://doi.org/10.48550/arXiv.1910.02798](https://doi.org/10.48550)]"
    "(https://doi.org/10.48550)<br>Focus to learn more<br>arXiv-issued DOI via DataCite"
)

OPTICA_TITLE = (
    "# All-Fiber Wavefront Shaping by Transmission Matrix Engineering ##### Author "
    "Affiliations - [Email](mailto:?subject=Article%20Published) - Share - "
    "[Share with Facebook](https://www.facebook.com/sharer.php?u=x) - "
    "[![Add to BibSonomy](https://opg.optica.org/images/bibsonomy-icon.png)Add to "
    "BibSonomy](https://www.bibsonomy.org/Show)"
)

LAB_TITLE = (
    "Wavefront shaping in multimode fibers by transmission matrix engineering | "
    "Complex Photonics Lab ## Filter by year - [2025](https://x/year/2025) (9) - "
    "[2024](https://x/2024) (9) - [2023](https://x/2023) (3)"
)


def test_arxiv_metadata_table_reduces_to_the_paper_title():
    assert to_label(ARXIV_TITLE) == (
        "Wavefront shaping in multimode fibers by transmission matrix engineering"
    )


def test_publisher_share_rail_reduces_to_the_paper_title():
    assert to_label(OPTICA_TITLE) == (
        "All-Fiber Wavefront Shaping by Transmission Matrix Engineering"
    )


def test_site_furniture_after_a_pipe_is_dropped():
    assert to_label(LAB_TITLE) == (
        "Wavefront shaping in multimode fibers by transmission matrix engineering"
    )


def test_a_plain_snippet_survives_unchanged_up_to_the_limit():
    snippet = "We present a new approach for shaping light at the output of a fiber."
    assert to_label(snippet) == snippet


def test_a_navigation_rail_yields_nothing_at_all():
    # "" is the signal to omit the field rather than show the reader debris.
    assert to_label("- Share - [Email](mailto:?subject=x) - [![i](https://a.png)](https://c)") == ""


@pytest.mark.parametrize("junk", ["| | | |-|-| | | |", "", None, "   ", "###", "- - -"])
def test_strings_with_no_readable_content_yield_nothing(junk):
    assert to_label(junk) == ""


def test_a_hyphenated_title_is_not_cut_at_the_hyphen():
    assert to_label("Adam - A Method for Stochastic Optimization") == (
        "Adam - A Method for Stochastic Optimization"
    )


def test_long_text_is_truncated_on_a_word_boundary():
    out = to_label("word " * 60, limit=40)

    assert out.endswith("…")
    assert len(out) <= 41
    assert "wor…" not in out, "cut mid-word"


def test_html_line_breaks_do_not_survive_into_the_label():
    assert "<br>" not in to_label("A real title<br>Focus to learn more<br>and more text")

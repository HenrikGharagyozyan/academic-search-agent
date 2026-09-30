"""A Wikipedia formula reaches the model once, as TeX.

The strings below are copied from passages of a live run on "what is entropy".
Each formula arrives three times over — flattened MathML, an escaped TeX
annotation, a fallback image — and the flattened copy comes first: "S=kBln⁡Ω"
for Boltzmann's formula, a bare "θ" glued to the word before it.
"""

from unittest.mock import MagicMock

from app.application.agents.nodes import retrieve_and_chunk_node
from app.domain.search import ScrapedPage, SearchResult
from app.domain.text.maths import collapse_wiki_maths

SVG = "https://wikimedia.org/api/rest_v1/media/math/render/svg"

BOLTZMANN = (
    'to the [natural logarithm](https://en.wikipedia.org/wiki/Natural_logarithm "Natural logarithm") of this number:\n'
    "S=kBln\u2061Ω{\\\\displaystyle S=k\\_{\\\\text{B}}\\\\ln \\\\Omega }"
    "![{\\displaystyle S=k_{\\text{B}}\\ln \\Omega }]"
    f"({SVG}/ac0dd0415d44dd3dc839b6e4fef85e7696de8d39)\n"
    "The proportionality constant _k_ B is one of the fundamental constants of physics"
)

THETA = (
    'for the change in any [extensive quantity](https://en.wikipedia.org/wiki/Extensive_quantity "Extensive quantity")'
    "θ{\\\\textstyle \\\\theta }![{\\textstyle \\theta }]"
    f"({SVG}/a11744bd71a5eb6efe4f28e12ca57f874d82658c)"
    " in a thermodynamic system"
)

RATE = (
    "balance expression states that "
    "dθ/dt{\\\\textstyle \\\\mathrm {d} \\\\theta /\\\\mathrm {d} t}"
    "![{\\textstyle \\mathrm {d} \\theta /\\mathrm {d} t}]"
    f"({SVG}/d6033adeccedd6f37b1e20e1707bcf1cfd554ca9)"
    ", i.e. the rate of change of "
    "θ{\\\\textstyle \\\\theta }![{\\textstyle \\theta }]"
    f"({SVG}/a11744bd71a5eb6efe4f28e12ca57f874d82658c)"
)


def test_a_formula_on_its_own_line_becomes_display_maths():
    out = collapse_wiki_maths(BOLTZMANN)

    assert "of this number:\n$$S=k_{\\text{B}}\\ln \\Omega$$\nThe proportionality" in out


def test_the_flattened_copy_and_the_image_are_gone():
    out = collapse_wiki_maths(BOLTZMANN)

    assert "S=kBln" not in out
    assert "displaystyle" not in out
    assert "wikimedia.org" not in out


def test_an_inline_symbol_is_separated_from_the_link_it_was_glued_to():
    out = collapse_wiki_maths(THETA)

    assert out == (
        'for the change in any [extensive quantity](https://en.wikipedia.org/wiki/Extensive_quantity "Extensive quantity")'
        " $\\theta$ in a thermodynamic system"
    )


def test_two_formulas_in_one_sentence_are_each_collapsed():
    out = collapse_wiki_maths(RATE)

    assert out == (
        "balance expression states that $\\mathrm {d} \\theta /\\mathrm {d} t$"
        ", i.e. the rate of change of $\\theta$"
    )


def test_the_symbol_itself_is_never_lost():
    # The report was a stray word "theta" in an answer; whatever else happens,
    # cleaning must not be where a symbol disappears.
    for passage in (BOLTZMANN, THETA, RATE):
        out = collapse_wiki_maths(passage)
        assert "\\Omega" in out or "\\theta" in out


def test_ordinary_text_and_ordinary_images_are_left_alone():
    text = "Entropy rises. ![A diagram](https://upload.wikimedia.org/x.png) Ω and θ stay."

    assert collapse_wiki_maths(text) == text


def test_pages_are_cleaned_before_they_are_cut_into_chunks():
    search = MagicMock()
    search.scrape.return_value = ScrapedPage(
        url="https://en.wikipedia.org/wiki/Entropy", title="Entropy", markdown=BOLTZMANN
    )
    state = {
        "search_results": [
            SearchResult(title="Entropy", url="https://en.wikipedia.org/wiki/Entropy", snippet="")
        ],
    }

    out = retrieve_and_chunk_node(state, search_provider=search)

    text = "\n".join(c.text for c in out["chunks"])
    assert "$$S=k_{\\text{B}}\\ln \\Omega$$" in text
    assert "S=kBln" not in text

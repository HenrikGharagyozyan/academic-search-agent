"""Authors and year come from the page's own citation tags.

The metadata below is what Firecrawl returned for two real sources on
transmission matrix engineering: the arXiv preprint of Resisi et al. and the
same group's Frontiers in Optics paper a year later.
"""

from app.domain.citation import (
    SourceCitation,
    citation_from_metadata,
    family_name,
    short_attribution,
)

ARXIV = {
    "title": "[1910.02798] Wavefront shaping in multimode fibers by transmission matrix engineering",
    "citation_online_date": "2019/10/07",
    "citation_date": "2019/10/07",
    "citation_author": ["Resisi, Shachar", "Viernik, Yehonatan", "Popoff, Sebastien", "Bromberg, Yaron"],
    "published_time": None,
}

OPTICA = {
    "dc_date": "2020-09-14",
    "dc.date": "2020-09-14",
    "citation_author": ["Shachar Resisi", "Sebastien M. Popoff", "Yaron Bromberg"],
    "citation_publication_date": "2020/09/14",
    "citation_start_date": "2020/09/14",
}


def test_an_arxiv_page_gives_its_authors_and_year():
    citation = citation_from_metadata(ARXIV)

    assert citation.year == 2019
    assert [family_name(a) for a in citation.authors] == ["Resisi", "Viernik", "Popoff", "Bromberg"]


def test_a_publisher_page_gives_its_publication_year():
    citation = citation_from_metadata(OPTICA)

    assert citation.year == 2020
    assert [family_name(a) for a in citation.authors] == ["Resisi", "Popoff", "Bromberg"]


def test_the_publication_date_outranks_a_generic_page_date():
    citation = citation_from_metadata(
        {"published_time": "2025-01-02", "citation_publication_date": "2021/05/01"}
    )

    assert citation.year == 2021


def test_a_page_that_declares_nothing_has_no_citation():
    assert citation_from_metadata({"title": "A blog post"}) == SourceCitation()


def test_a_single_author_string_is_accepted():
    assert citation_from_metadata({"citation_author": "Jiang, Wei"}).authors == ["Jiang, Wei"]


def test_the_short_form_follows_the_number_of_authors():
    assert short_attribution(citation_from_metadata(ARXIV)) == "Resisi et al. (2019)"
    assert short_attribution(SourceCitation(authors=["A. Cheng", "B. Li"], year=2023)) == "Cheng and Li (2023)"
    assert short_attribution(SourceCitation(authors=["Cheng, A."])) == "Cheng"
    assert short_attribution(SourceCitation(year=2023)) == ""

"""Deduplicating by content rather than by URL, and keeping one source from
taking the whole context.

The reported case: six results from six domains that were one paper — the arXiv
version, the publisher's, the lab's publication page, two aggregator stubs.
Deduplicating by URL sees six sources. Comparing chunk text exactly sees six
too, because each site wraps the same words in its own furniture.
"""

from unittest.mock import MagicMock

import pytest

from app.application.agents.nodes import retrieve_and_chunk_node, select_relevant_chunks_node
from app.domain.documents import Chunk
from app.domain.selection import cap_per_source, source_count
from app.domain.text.similarity import MIRROR_THRESHOLD, containment, find_mirrors, shingles
from app.domain.search import ScrapedPage, SearchResult

ABSTRACT = (
    "We present a new approach for shaping light at the output of a multimode fiber by "
    "modulating the transmission matrix of the system rather than the incident wavefront. "
    "We show that by applying controlled perturbations to the fiber we can tune the "
    "transmission matrix and focus light at the fiber output."
)

# The publisher's page: the same abstract, plus the method and results.
ARTICLE = ABSTRACT + " " + (
    "Multimode fibers support many propagation modes and the transmission matrix relates "
    "input and output fields. We measured the matrix using off-axis holography and applied "
    "macro-bending perturbations at nine positions along the fiber. "
) * 6

# An aggregator: the same abstract wrapped in site furniture.
STUB = "Home Publications Search " + ABSTRACT + " Cite Share Download PDF Related articles"

# A genuinely different paper on the same topic: shared vocabulary, not phrasing.
OTHER = (
    "Online learning of the transmission matrix enables real-time correction in dynamically "
    "changing multimode fibers. We train a neural network on intensity-only measurements and "
    "update the estimate as the fiber bends, achieving reference-free retrieval. "
) * 5


def chunk(source: str, index: int = 0, text: str = "passage text") -> Chunk:
    return Chunk(
        chunk_id=f"{source}#{index}", document_id=source, text=text,
        start_line=1, end_line=1, source_url=source, title="T",
    )


# --- the similarity measure ------------------------------------------------


def test_a_mirror_scores_far_above_the_threshold():
    score = containment(shingles(ARTICLE), shingles(STUB))

    assert score > MIRROR_THRESHOLD
    assert score > 0.8, f"mirrors should be unmistakable, got {score:.2f}"


def test_two_different_papers_on_one_topic_score_far_below():
    """Jaccard would be the wrong measure here and this is the reason to check:
    a shared vocabulary produces almost no shared five-word runs."""
    score = containment(shingles(ARTICLE), shingles(OTHER))

    assert score < MIRROR_THRESHOLD
    assert score < 0.2, f"distinct papers should not look alike, got {score:.2f}"


def test_containment_not_jaccard_catches_an_abstract_inside_an_article():
    # The abstract is a fifth of the article's length; Jaccard would score that
    # pair low precisely when one document wholly contains the other.
    assert containment(shingles(ARTICLE), shingles(ABSTRACT)) == pytest.approx(1.0)


def test_an_empty_document_matches_nothing():
    assert containment(shingles(""), shingles(ARTICLE)) == 0.0


def test_find_mirrors_keeps_the_fullest_version():
    texts = [STUB, ARTICLE, ABSTRACT]

    mirrors = find_mirrors(texts)

    # Index 1 is the article: the longest, so it is the one kept.
    assert mirrors == {0: 1, 2: 1}


def test_find_mirrors_leaves_distinct_documents_alone():
    assert find_mirrors([ARTICLE, OTHER]) == {}


def test_find_mirrors_is_stable_for_equally_long_documents():
    assert find_mirrors([ABSTRACT, ABSTRACT]) == {1: 0}


# --- the retrieval node ----------------------------------------------------


def test_mirrored_pages_are_dropped_before_anything_is_embedded():
    pages = {
        "https://arxiv.org/abs/1": ABSTRACT,
        "https://publisher.com/full": ARTICLE,
        "https://aggregator.com/stub": STUB,
        "https://other-lab.org/paper": OTHER,
    }
    provider = MagicMock()
    provider.scrape.side_effect = lambda url: ScrapedPage(
        url=url, title="A paper", markdown=pages[url]
    )
    state = {
        "search_results": [
            SearchResult(title="t", url=url, snippet="...") for url in pages
        ]
    }

    out = retrieve_and_chunk_node(state, provider)
    sources = {c.source_url for c in out["chunks"]}

    # One paper survives as one source, and the different paper survives too.
    assert sources == {"https://publisher.com/full", "https://other-lab.org/paper"}
    dropped = [s for s in out["activity"] if s.kind == "mirror_dropped"]
    assert len(dropped) == 2
    assert all(s.url in pages for s in dropped)


def test_the_trail_says_which_page_a_mirror_was_a_copy_of():
    pages = {"https://a.com": ARTICLE, "https://b.com": STUB}
    provider = MagicMock()
    provider.scrape.side_effect = lambda url: ScrapedPage(url=url, title="t", markdown=pages[url])
    state = {"search_results": [SearchResult(title="t", url=u, snippet="") for u in pages]}

    out = retrieve_and_chunk_node(state, provider)
    step = next(s for s in out["activity"] if s.kind == "mirror_dropped")

    assert "b.com" in step.label and "a.com" in step.label
    assert step.detail == "kept the fuller version"


def test_distinct_pages_are_all_kept():
    pages = {"https://a.com": ARTICLE, "https://b.com": OTHER}
    provider = MagicMock()
    provider.scrape.side_effect = lambda url: ScrapedPage(url=url, title="t", markdown=pages[url])
    state = {"search_results": [SearchResult(title="t", url=u, snippet="") for u in pages]}

    out = retrieve_and_chunk_node(state, provider)

    assert {c.source_url for c in out["chunks"]} == set(pages)
    assert not [s for s in out["activity"] if s.kind == "mirror_dropped"]


# --- the per-source quota --------------------------------------------------


def test_one_source_cannot_take_every_slot():
    ranked = [chunk("https://dominant.com", i) for i in range(20)]
    ranked += [chunk("https://other.com", i) for i in range(20)]

    taken = cap_per_source(ranked, limit=10, max_per_source=4)

    assert len(taken) == 10
    assert source_count(taken) == 2


def test_the_quota_does_not_leave_the_context_short():
    """Running short would trade one bias for a worse one: less evidence."""
    ranked = [chunk("https://only.com", i) for i in range(20)]

    taken = cap_per_source(ranked, limit=10, max_per_source=4)

    assert len(taken) == 10, "a single-source result set must still fill the budget"


def test_rank_order_is_preserved():
    ranked = [chunk("https://a.com", 0), chunk("https://b.com", 0), chunk("https://a.com", 1)]

    taken = cap_per_source(ranked, limit=3, max_per_source=2)

    assert [c.chunk_id for c in taken] == [c.chunk_id for c in ranked]


def test_fewer_chunks_than_the_limit_are_all_returned():
    ranked = [chunk("https://a.com", 0), chunk("https://b.com", 0)]

    assert cap_per_source(ranked, limit=10, max_per_source=5) == list(ranked)


@pytest.mark.parametrize(("limit", "quota"), [(0, 5), (5, 0), (0, 0)])
def test_a_zero_budget_selects_nothing(limit, quota):
    assert cap_per_source([chunk("https://a.com", 0)], limit, quota) == []


def test_the_selection_node_oversamples_then_applies_the_quota():
    from app.application.agents.constants import (
        CHUNK_OVERSAMPLE,
        MAX_SOURCE_SHARE,
        TOP_K_CHUNKS,
    )

    chunks = [chunk("https://dominant.com", i) for i in range(60)]
    chunks += [chunk("https://other.com", i) for i in range(60)]
    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda q, cs, top_k: cs[:top_k]

    out = select_relevant_chunks_node({"question": "q?", "chunks": chunks}, store)

    # It asked for more than it needs, because the quota discards some.
    assert store.select_relevant_chunks.call_args.kwargs["top_k"] == (
        TOP_K_CHUNKS * CHUNK_OVERSAMPLE
    )
    assert len(out["selected_chunks"]) == TOP_K_CHUNKS
    per_source = {}
    for c in out["selected_chunks"]:
        per_source[c.source_url] = per_source.get(c.source_url, 0) + 1
    # Without the quota the dominant source would hold all 25 slots.
    assert max(per_source.values()) < TOP_K_CHUNKS
    assert source_count(out["selected_chunks"]) == 2
    assert MAX_SOURCE_SHARE < 1.0


# --- the case content similarity alone missed ------------------------------

TITLE = "Wavefront shaping in multimode fibers by transmission matrix engineering"

# What arXiv's abstract page actually looks like once scraped: the abstract is a
# small part of it, the rest is subject tags, submission history and links to
# bibliographic tools.
ARXIV_PAGE = (
    "arXiv Subjects Optics physics.optics Quantum Physics quant-ph Cite as arXiv 1910.02798 "
    "Submission history From Sebastien Popoff view email Bibliographic Tools Bibliographic "
    "Explorer Toggle Connected Papers Toggle Litmaps Toggle scite Smart Citations Code Data "
    "Media Demos Related Papers About arXivLabs Which authors of this paper are endorsers "
    "Disable MathJax What is MathJax Browse context new recent "
) * 8 + ABSTRACT

# The publisher's copy: the same abstract plus the whole paper. A real article
# dwarfs the preprint listing page, which is what makes the containment fail —
# the shared abstract is a small fraction of the arXiv page's own text.
FULL_ARTICLE = ABSTRACT + " " + (
    "Multimode fibers support many propagation modes and the transmission matrix relates "
    "input and output fields. We measured the matrix using off-axis holography and applied "
    "macro-bending perturbations at nine positions along the fiber, recording the resulting "
    "intensity patterns for each configuration of the mechanical actuators. "
) * 60


def test_content_alone_does_not_catch_an_arxiv_page_against_the_article():
    """Documents the limit that made the title signal necessary. A live run on
    the reported question kept both the arXiv page and the publisher's copy of
    the same paper, because this score sits below the threshold."""
    score = containment(shingles(ARXIV_PAGE), shingles(FULL_ARTICLE))

    assert score < MIRROR_THRESHOLD, (
        f"if content alone now catches this, the furniture assumption changed ({score:.2f})"
    )
    assert find_mirrors([ARXIV_PAGE, FULL_ARTICLE]) == {}, "content alone should miss it"


def test_an_identical_title_settles_it_where_content_could_not():
    mirrors = find_mirrors([ARXIV_PAGE, FULL_ARTICLE], titles=[TITLE, TITLE])

    # The article is the longer document, so the listing page is the one dropped.
    assert len(FULL_ARTICLE) > len(ARXIV_PAGE)
    assert mirrors == {0: 1}


def test_the_fuller_version_is_kept_whichever_order_they_arrive_in():
    """Which is the point: the publisher's article has the method and the
    figures that the preprint listing page does not."""
    assert find_mirrors([FULL_ARTICLE, ARXIV_PAGE], titles=[TITLE, TITLE]) == {1: 0}
    assert find_mirrors([ARXIV_PAGE, FULL_ARTICLE], titles=[TITLE, TITLE]) == {0: 1}


def test_a_generic_title_does_not_merge_unrelated_pages():
    """"Adaptive Optics" appeared in the same live run both as a reference page
    and inside a paper's title. They are not the same document."""
    mirrors = find_mirrors([OTHER, FULL_ARTICLE], titles=["Adaptive Optics", "Adaptive Optics"])

    assert mirrors == {}, "a two-word title must not be treated as an identity"


def test_different_titles_on_the_same_topic_are_not_merged():
    mirrors = find_mirrors(
        [OTHER, FULL_ARTICLE],
        titles=[
            "Online learning of the transmission matrix for real-time correction",
            "Wavefront shaping in multimode fibers by transmission matrix engineering",
        ],
    )

    assert mirrors == {}


def test_pages_with_unusable_titles_still_get_the_content_comparison():
    assert find_mirrors([FULL_ARTICLE, STUB], titles=["", ""]) == {1: 0}


def test_sibilant_nouns_are_pluralised_correctly():
    """"Planned 4 searchs" reached a live progress line."""
    from app.application.agents.activity import count

    assert count(4, "search") == "4 searches"
    assert count(1, "search") == "1 search"
    assert count(2, "passage") == "2 passages"
    assert count(3, "box") == "3 boxes"

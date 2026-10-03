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
from app.domain.selection import source_count, spread_across_sources
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


# --- spreading the context across sources ----------------------------------


def test_every_source_gets_a_turn_before_any_page_repeats():
    ranked = [chunk("https://long.com", i) for i in range(100)]
    ranked += [chunk("https://short.com", i) for i in range(2)]

    taken = spread_across_sources(ranked, limit=10)

    assert source_count(taken) == 2
    assert [c.source_url for c in taken[:2]] == ["https://long.com", "https://short.com"]


def test_the_shape_that_defeated_a_per_source_quota():
    """The live run this replaced: one page held 114 of 133 passages, so the top
    of the ranking was entirely its own and a 10-of-25 cap never came into play.
    All 25 slots went to one source."""
    ranked = [chunk("https://pubs.aip.org", i) for i in range(114)]
    ranked += [chunk("https://link.aps.org", i) for i in range(13)]
    ranked += [chunk("https://opg.optica.org", i) for i in range(10)]
    ranked += [chunk("https://hal.science", i) for i in range(1)]

    taken = spread_across_sources(ranked, limit=25)

    assert len(taken) == 25
    assert source_count(taken) == 4, "every page that was read must be represented"
    # And no page holds most of the context any more.
    per_source = {}
    for c in taken:
        per_source[c.source_url] = per_source.get(c.source_url, 0) + 1
    assert max(per_source.values()) <= 10


def test_the_best_ranked_source_still_leads():
    ranked = [chunk("https://best.com", 0), chunk("https://second.com", 0)]

    taken = spread_across_sources(ranked, limit=2)

    assert taken[0].source_url == "https://best.com"


def test_rank_order_is_preserved_within_a_source():
    ranked = [chunk("https://a.com", i) for i in range(3)]

    taken = spread_across_sources(ranked, limit=3)

    assert [c.chunk_id for c in taken] == ["https://a.com#0", "https://a.com#1", "https://a.com#2"]


def test_a_long_page_fills_the_slots_the_others_cannot():
    """Breadth first, then depth: running short would trade one bias for a worse
    one, less evidence overall."""
    ranked = [chunk("https://long.com", i) for i in range(50)]
    ranked += [chunk("https://tiny.com", 0)]

    taken = spread_across_sources(ranked, limit=10)

    assert len(taken) == 10
    assert sum(1 for c in taken if c.source_url == "https://long.com") == 9


def test_a_single_source_still_fills_the_budget():
    ranked = [chunk("https://only.com", i) for i in range(20)]

    assert len(spread_across_sources(ranked, limit=10)) == 10


def test_fewer_passages_than_the_limit_are_all_returned():
    ranked = [chunk("https://a.com", 0), chunk("https://b.com", 0)]

    assert spread_across_sources(ranked, limit=10) == list(ranked)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_zero_budget_selects_nothing(limit):
    assert spread_across_sources([chunk("https://a.com", 0)], limit) == []


def test_the_selection_node_ranks_everything_then_spreads_it():
    from app.application.agents.constants import TOP_K_CHUNKS

    chunks = [chunk("https://dominant.com", i) for i in range(100)]
    chunks += [chunk("https://other.com", i) for i in range(4)]
    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda q, cs, top_k: cs[:top_k]

    out = select_relevant_chunks_node({"question": "q?", "chunks": chunks}, store)

    # The whole set is ranked: truncating first is what hid the small sources.
    assert store.select_relevant_chunks.call_args.kwargs["top_k"] == len(chunks)
    assert len(out["selected_chunks"]) == TOP_K_CHUNKS
    assert source_count(out["selected_chunks"]) == 2
    assert "each source gets a turn" in out["activity"][0].detail


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


# --- the provider's own rate limit -----------------------------------------


def test_a_rate_limited_scrape_is_retried_then_succeeds():
    """Twelve concurrent scrapes exhausted the per-minute budget, and the pages
    it refused were reported as unreadable — so a quota problem looked like a
    broken site and which sources an answer used varied run to run."""
    from unittest.mock import patch

    from app.infrastructure.search.firecrawl import FirecrawlProvider

    with patch("app.infrastructure.search.firecrawl.FirecrawlApp") as app_cls:
        client = MagicMock()
        calls = {"n": 0}

        def scrape(url, formats):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Rate Limit Exceeded: Consumed (req/min): 11")
            page = MagicMock()
            page.markdown = "content"
            page.metadata.title = "T"
            return page

        client.scrape.side_effect = scrape
        app_cls.return_value = client

        provider = FirecrawlProvider()
        with patch("app.infrastructure.search.firecrawl.time.sleep"):
            page = provider.scrape("https://example.com")

    assert page.markdown == "content"
    assert calls["n"] == 2


def test_a_persistent_rate_limit_is_reported_as_itself():
    from unittest.mock import patch

    from app.infrastructure.search.firecrawl import (
        RATE_LIMIT_RETRIES,
        FirecrawlProvider,
        ScrapeRateLimited,
    )

    with patch("app.infrastructure.search.firecrawl.FirecrawlApp") as app_cls:
        client = MagicMock()
        client.scrape.side_effect = RuntimeError("Rate limit exceeded")
        app_cls.return_value = client

        provider = FirecrawlProvider()
        with patch("app.infrastructure.search.firecrawl.time.sleep"), pytest.raises(
            ScrapeRateLimited
        ):
            provider.scrape("https://example.com")

        assert client.scrape.call_count == RATE_LIMIT_RETRIES + 1


def test_an_ordinary_scrape_failure_is_not_retried():
    """Retrying a page that simply cannot be read wastes the quota that the
    retry exists to protect."""
    from unittest.mock import patch

    from app.infrastructure.search.firecrawl import FirecrawlProvider

    with patch("app.infrastructure.search.firecrawl.FirecrawlApp") as app_cls:
        client = MagicMock()
        client.scrape.side_effect = RuntimeError("Website Not Supported")
        app_cls.return_value = client

        provider = FirecrawlProvider()
        with pytest.raises(RuntimeError):
            provider.scrape("https://example.com")

        assert client.scrape.call_count == 1


def test_the_trail_tells_a_rate_limit_apart_from_a_broken_page():
    from app.infrastructure.search.firecrawl import ScrapeRateLimited

    provider = MagicMock()

    def scrape(url):
        if "limited" in url:
            raise ScrapeRateLimited("quota")
        if "broken" in url:
            raise RuntimeError("Website Not Supported")
        return ScrapedPage(url=url, title="t", markdown=ARTICLE)

    provider.scrape.side_effect = scrape
    state = {
        "search_results": [
            SearchResult(title="t", url=u, snippet="")
            for u in ("https://limited.com", "https://broken.com", "https://fine.com")
        ]
    }

    out = retrieve_and_chunk_node(state, provider)
    failures = {s.url: s for s in out["activity"] if s.kind == "scrape_failed"}

    assert "rate limit" in failures["https://limited.com"].label
    assert failures["https://limited.com"].detail == "not a problem with the page"
    assert "Could not read" in failures["https://broken.com"].label
    assert failures["https://broken.com"].detail is None

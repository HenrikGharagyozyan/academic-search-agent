"""The activity trail: what the agent reports while it works, and what it
leaves behind on the answer.

The two properties worth pinning are that a slow node reports *during* its run
rather than only on return, and that the trail accumulates instead of being
overwritten — a refine pass must add to the record of the first attempt, not
erase it.
"""

from unittest.mock import MagicMock

import pytest

from app.application.agents.activity import ActivityRecorder, short_host
from app.application.agents.nodes import (
    retrieve_and_chunk_node,
    search_node,
    verify_evidence_node,
)
from app.application.services.research import ResearchService
from app.core.exceptions import UpstreamServiceError
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.search import ScrapedPage, SearchResult


def result(url: str, title: str = "A paper") -> SearchResult:
    return SearchResult(title=title, url=url, snippet="...")


def page(url: str, body: str = "Gradient descent converges for convex functions.") -> ScrapedPage:
    return ScrapedPage(url=url, title="A paper", markdown=body)


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc", text="text",
        start_line=1, end_line=1, source_url="https://example.com", title="T",
    )


# --- recorder --------------------------------------------------------------


def test_recorder_collects_steps_without_a_graph_around_it():
    # Nodes are unit-tested by direct call, where there is no run to stream to.
    recorder = ActivityRecorder()
    recorder.record("search", "Searching", detail="q")

    assert [s.kind for s in recorder.steps] == ["search"]


def test_recorder_stamps_the_refine_pass_on_every_step():
    recorder = ActivityRecorder(attempt=2)
    recorder.record("search", "Searching again")

    assert recorder.steps[0].attempt == 2


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.nature.com/articles/x", "nature.com"),
        ("https://arxiv.org/abs/1706.03762", "arxiv.org"),
        ("https://en.wikipedia.org/wiki/Adam", "en.wikipedia.org"),
    ],
)
def test_short_host_drops_the_noise(url, expected):
    assert short_host(url) == expected


# --- per-source reporting --------------------------------------------------


def test_search_records_the_query_and_every_source_it_found():
    provider = MagicMock()
    provider.search.return_value = [
        result("https://arxiv.org/abs/1"),
        result("https://www.nature.com/articles/2"),
    ]

    out = search_node({"question": "q?", "search_query": "convex convergence"}, provider)
    steps = out["activity"]

    assert steps[0].kind == "search"
    assert "convex convergence" in steps[0].label
    assert steps[0].detail is None, "the label already carries the query"
    assert [s.kind for s in steps[1:]] == ["source_found", "source_found"]
    assert [s.url for s in steps[1:]] == [
        "https://arxiv.org/abs/1",
        "https://www.nature.com/articles/2",
    ]


def test_search_records_nothing_extra_when_the_provider_fails():
    provider = MagicMock()
    provider.search.side_effect = RuntimeError("boom")

    with pytest.raises(UpstreamServiceError):
        search_node({"question": "q?", "search_query": "q"}, provider)


def test_retrieval_reports_one_step_per_page_read():
    provider = MagicMock()
    provider.scrape.side_effect = lambda url: page(url)
    state = {
        "search_results": [
            result("https://a.com"), result("https://b.com"), result("https://c.com")
        ]
    }

    out = retrieve_and_chunk_node(state, provider)
    read = [s for s in out["activity"] if s.kind == "scrape_ok"]

    assert len(read) == 3
    assert {s.url for s in read} == {"https://a.com", "https://b.com", "https://c.com"}
    # Each one says how much it got, which is the point of a per-page line.
    assert all(s.detail and "passage" in s.detail for s in read)


def test_retrieval_reports_a_page_it_could_not_read():
    provider = MagicMock()

    def scrape(url):
        if url == "https://broken.com":
            raise RuntimeError("Website Not Supported")
        return page(url)

    provider.scrape.side_effect = scrape
    state = {"search_results": [result("https://broken.com"), result("https://ok.com")]}

    out = retrieve_and_chunk_node(state, provider)
    kinds = {s.kind: s for s in out["activity"] if s.kind in {"scrape_ok", "scrape_failed"}}

    assert kinds["scrape_failed"].url == "https://broken.com"
    assert kinds["scrape_ok"].url == "https://ok.com"
    # A page that cannot be read is reported, not silently missing.
    assert "Could not read" in kinds["scrape_failed"].label


def test_retrieval_keeps_chunks_in_search_order_not_completion_order():
    """Scrapes finish in whatever order the network decides, but the vector
    store falls back to the first top_k chunks when embedding fails — so the
    order of `chunks` decides what survives and must stay deterministic."""
    provider = MagicMock()
    provider.scrape.side_effect = lambda url: page(url, f"Body of {url} with enough text here.")
    state = {
        "search_results": [result("https://first.com"), result("https://second.com")]
    }

    out = retrieve_and_chunk_node(state, provider)
    sources = [c.source_url for c in out["chunks"]]

    assert sources.index("https://first.com") < sources.index("https://second.com")


def test_verify_reports_what_it_threw_away():
    state = {
        "selected_chunks": [chunk("real")],
        "claims": [
            Claim(text="grounded", evidence_ids=["real", "invented"], confidence="high"),
            Claim(text="ungrounded", evidence_ids=["invented"], confidence="low"),
        ],
    }

    out = verify_evidence_node(state)
    step = out["activity"][0]

    assert step.kind == "verify"
    assert "1 claim dropped as ungrounded" in step.detail
    assert "1 invented citation removed" in step.detail

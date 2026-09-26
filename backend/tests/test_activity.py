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

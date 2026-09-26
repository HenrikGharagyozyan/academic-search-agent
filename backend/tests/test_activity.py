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


# --- the trail on a finished answer ---------------------------------------


@pytest.fixture
def pipeline(keep_all_chunks_relevant, answer_is_satisfactory):
    search = MagicMock()
    search.search.return_value = [result("https://arxiv.org/abs/1")]
    search.scrape.side_effect = lambda url: page(url)

    llm = MagicMock()
    llm.grade_relevance.side_effect = keep_all_chunks_relevant
    llm.grade_answer_quality.return_value = answer_is_satisfactory
    llm.generate_answer.side_effect = lambda q, evidence: ClaimsResponse(
        summary="S",
        claims=[Claim(text="C", evidence_ids=[evidence[0].chunk_id], confidence="high")],
        conclusion="K",
    )

    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda q, chunks, top_k=15: chunks

    return ResearchService(search_provider=search, llm=llm, vector_store=store), search, llm


def test_answer_carries_the_trail(pipeline):
    service, _, _ = pipeline

    answer = service.answer("Does gradient descent converge?")
    kinds = [s.kind for s in answer.activity]

    # The whole run is on the answer, so the panel works after a reload and for
    # a caller that never streamed.
    for expected in ("search", "source_found", "scrape_ok", "select",
                     "grade_relevance", "generate", "verify", "grade_answer"):
        assert expected in kinds, f"{expected} missing from {kinds}"


def test_trail_survives_a_refine_pass(pipeline, keep_all_chunks_relevant):
    """`activity` is the first field in ResearchState with a reducer. Without
    one, each node's list would replace the previous node's instead of adding
    to it, and a second search would erase the record of the first."""
    service, _, llm = pipeline
    llm.refine_query.return_value = "a better query"

    calls = {"n": 0}

    def generate(question, evidence):
        calls["n"] += 1
        if calls["n"] == 1:
            return ClaimsResponse(
                summary="", claims=[Claim(text="x", evidence_ids=["nope"], confidence="low")],
                conclusion="",
            )
        return ClaimsResponse(
            summary="S",
            claims=[Claim(text="C", evidence_ids=[evidence[0].chunk_id], confidence="high")],
            conclusion="K",
        )

    llm.generate_answer.side_effect = generate

    answer = service.answer("q?")
    attempts = {s.attempt for s in answer.activity}

    assert attempts == {0, 1}, "both passes must be on the trail"
    assert [s.kind for s in answer.activity].count("search") == 2
    assert any(s.kind == "refine" for s in answer.activity)


def test_stream_emits_activity_before_the_result(pipeline):
    service, _, _ = pipeline

    events = list(service.stream_answer("q?"))
    activity = [e for e in events if e["event"] == "activity"]

    assert activity, "no activity reached the stream"
    assert events[-1]["event"] == "result"
    # Live steps are the same shape as the ones stored on the answer.
    assert {"kind", "label", "attempt"} <= set(activity[0]["data"])


def test_per_page_steps_arrive_before_the_node_finishes(pipeline):
    """The reason for streaming these at all: scraping is one node that can run
    for most of a minute, so its per-page steps must reach the client before the
    node's own stage event does."""
    service, _, _ = pipeline

    events = list(service.stream_answer("q?"))
    order = [
        (e["event"], e["data"].get("kind") or e["data"].get("stage"))
        for e in events
        if e["event"] in {"activity", "progress"}
    ]

    scrape_step = next(i for i, (ev, k) in enumerate(order) if k == "scrape_ok")
    retrieval_stage = next(
        i for i, (ev, k) in enumerate(order) if ev == "progress" and k == "retrieve_and_chunk"
    )

    assert scrape_step < retrieval_stage

"""The streaming endpoint had no coverage at all, and its worst failure is
silent: once the response headers are out, an exception cannot become a 502 —
the connection just closes, and the browser shows an empty page. Every path
through it has to end in a terminal event."""

import json
from unittest.mock import MagicMock

import pytest

from app.application.services.research import ResearchService
from app.domain.answers import Claim, ClaimsResponse
from app.domain.search import ScrapedPage, SearchResult


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        name = next(l[len("event: "):] for l in lines if l.startswith("event: "))
        data = next(l[len("data: "):] for l in lines if l.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


@pytest.fixture
def pipeline(keep_all_chunks_relevant):
    """A research service wired to mocks, so the real graph actually runs."""
    search = MagicMock()
    search.search.return_value = [
        SearchResult(title="Paper", url="https://example.com", snippet="...")
    ]
    search.scrape.return_value = ScrapedPage(
        url="https://example.com", title="Paper", markdown="Gradient descent converges."
    )

    llm = MagicMock()
    llm.grade_relevance.side_effect = keep_all_chunks_relevant
    llm.generate_answer.side_effect = lambda question, evidence: ClaimsResponse(
        summary="Test summary",
        claims=[Claim(text="A claim", evidence_ids=[evidence[0].chunk_id], confidence="high")],
        conclusion="Test conclusion",
    )

    store = MagicMock()
    store.select_relevant_chunks.side_effect = lambda question, chunks, top_k=15: chunks

    return ResearchService(search_provider=search, llm=llm, vector_store=store), search, llm


def test_stream_reports_progress_then_a_result(pipeline):
    service, _, _ = pipeline

    events = list(service.stream_answer("Does gradient descent converge?"))

    names = [e["event"] for e in events]
    assert names[-1] == "result"
    assert names[:-1] == ["progress"] * (len(names) - 1)
    assert events[-1]["data"]["summary"] == "Test summary"
    assert events[-1]["data"]["claims"][0]["text"] == "A claim"


def test_every_progress_event_carries_a_human_readable_label(pipeline):
    service, _, _ = pipeline

    for event in service.stream_answer("q?"):
        if event["event"] == "progress":
            assert event["data"]["label"]
            assert event["data"]["label"] != event["data"]["stage"]


def test_an_upstream_failure_becomes_an_error_event(pipeline):
    service, search, _ = pipeline
    search.search.side_effect = RuntimeError("firecrawl is down")

    events = list(service.stream_answer("q?"))

    assert events[-1]["event"] == "error"
    assert "Search provider failed" in events[-1]["data"]["detail"]


def test_a_failure_while_building_the_answer_still_reaches_the_client(pipeline):
    """_build_answer used to sit outside the try block, so a fault there
    escaped the generator after the headers had gone out — no error event, no
    status code, just a truncated stream."""
    service, _, _ = pipeline
    service._build_answer = MagicMock(side_effect=RuntimeError("boom"))

    events = list(service.stream_answer("q?"))

    assert events[-1] == {"event": "error", "data": {"detail": "Research pipeline failed"}}


def test_an_internal_failure_does_not_leak_its_message(pipeline):
    service, _, _ = pipeline
    service._build_answer = MagicMock(side_effect=RuntimeError("secret internal detail"))

    events = list(service.stream_answer("q?"))

    assert "secret" not in json.dumps(events[-1])


# --- transport ------------------------------------------------------------


def test_endpoint_emits_well_formed_sse(client, mock_research_service):
    mock_research_service.stream_answer.return_value = iter([
        {"event": "progress", "data": {"stage": "search", "label": "Searching sources"}},
        {"event": "result", "data": {"question": "q?", "summary": "s"}},
    ])

    response = client.post("/api/v1/answer/stream", json={"question": "valid question"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Buffering proxies would hold the whole stream and defeat the point.
    assert response.headers["x-accel-buffering"] == "no"
    assert parse_sse(response.text) == [
        ("progress", {"stage": "search", "label": "Searching sources"}),
        ("result", {"question": "q?", "summary": "s"}),
    ]


def test_endpoint_encodes_newlines_so_they_cannot_split_an_event(client, mock_research_service):
    """A raw newline inside the payload would terminate the data field early
    and desynchronise the client's parser."""
    mock_research_service.stream_answer.return_value = iter([
        {"event": "result", "data": {"summary": "first line\nsecond line"}},
    ])

    response = client.post("/api/v1/answer/stream", json={"question": "valid question"})

    assert parse_sse(response.text) == [("result", {"summary": "first line\nsecond line"})]


def test_endpoint_rejects_an_invalid_question_before_streaming(client, mock_research_service):
    response = client.post("/api/v1/answer/stream", json={"question": "hi"})

    assert response.status_code == 422
    mock_research_service.stream_answer.assert_not_called()

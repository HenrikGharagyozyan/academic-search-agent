import logging
from concurrent.futures import ThreadPoolExecutor

from app.application.agents.activity import ActivityRecorder, count, short_host
from app.application.agents.constants import MAX_SOURCES, RESULTS_PER_QUERY
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.domain.search import SearchResult
from app.domain.text.plain import to_label
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


def _interleave(per_query: list[list[SearchResult]], limit: int) -> list[SearchResult]:
    """Takes results round-robin across queries, dropping repeated URLs.

    Round-robin rather than concatenation: the first query would otherwise fill
    the budget on its own and the other queries — the ones deliberately aimed
    elsewhere — would contribute nothing.
    """
    merged: list[SearchResult] = []
    seen: set[str] = set()

    for rank in range(max((len(results) for results in per_query), default=0)):
        for results in per_query:
            if len(merged) >= limit:
                return merged
            if rank >= len(results):
                continue
            result = results[rank]
            if result.url in seen:
                continue
            seen.add(result.url)
            merged.append(result)

    return merged


def search_node(state: ResearchState, search_provider: SearchProvider) -> dict:
    queries = state.get("search_queries") or [state.get("search_query") or state["question"]]
    since_year = state.get("since_year")
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    def run(query: str) -> tuple[str, list[SearchResult] | None]:
        try:
            return query, search_provider.search(
                query, limit=RESULTS_PER_QUERY, since_year=since_year
            )
        except Exception:
            logger.warning("Search failed for query=%r", query, exc_info=True)
            return query, None

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        outcomes = list(executor.map(run, queries))

    per_query: list[list[SearchResult]] = []
    for query, results in outcomes:
        if results is None:
            recorder.record("search", f"Search failed for “{query}”")
            continue
        per_query.append(results)
        recorder.record(
            "search",
            f"Searched for “{query}”",
            detail=count(len(results), "result"),
        )

    # Every query failing is an outage, not a topic with no coverage.
    if not per_query:
        raise UpstreamServiceError("Search provider failed for every query")

    results = _interleave(per_query, MAX_SOURCES)

    for result in results:
        # The engine hands back whatever it scraped: arXiv's metadata table, a
        # publisher's share rail. to_label keeps the readable opening of it and
        # returns "" when there is nothing readable, so the field is dropped
        # rather than filled with markup.
        name = to_label(result.title) or to_label(result.snippet)
        recorder.record(
            "source_found",
            f"Found {short_host(result.url)}",
            url=result.url,
            title=name or short_host(result.url),
            detail=name or None,
        )

    return {"search_results": results, "activity": recorder.steps}

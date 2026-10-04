import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.application.agents.activity import ActivityRecorder, count, short_host
from app.application.agents.constants import MAX_SOURCES, RESULTS_PER_QUERY
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.domain.search import SearchResult
from app.domain.sources import Candidate, select_sources
from app.domain.text.plain import to_label
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


_TIER_NAMES = {"scholarly": "scholarly", "reference": "reference"}


def _mix(chosen: list[Candidate]) -> str:
    """"7 scholarly · 3 reference · 2 other". Unknown and low-priority hosts are
    counted together on purpose: the reader is told what was favoured, not
    handed a verdict on a particular site."""
    tally = Counter(_TIER_NAMES.get(c.tier, "other") for c in chosen)
    return " · ".join(
        f"{tally[name]} {name}" for name in ("scholarly", "reference", "other") if tally[name]
    )


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

    # Every query failing is an outage, not a topic with no coverage — unless
    # this is a retry: an answer from the first pass is already in hand, and
    # an outage now should end the run on it rather than replace it with an
    # error.
    if not per_query:
        if state.get("kept_answer"):
            recorder.record("search", "Search failed for every query; keeping the earlier answer")
            return {"search_results": [], "activity": recorder.steps}
        raise UpstreamServiceError("Search provider failed for every query")

    # The question, not the planned queries: every query was planned from it, and
    # it is what the passages are ranked against later, so a page is judged by
    # the same measure at both ends.
    selection = select_sources(per_query, state["question"], MAX_SOURCES)
    results = [c.result for c in selection.chosen]

    details = [_mix(selection.chosen)]
    if selection.mirrors_set_aside:
        details.append(
            f"{count(selection.mirrors_set_aside, 'copy', 'copies')} of a chosen page set aside"
        )
    recorder.record(
        "rank",
        f"Ranked {count(selection.considered, 'candidate')}, reading {len(results)}",
        detail=", ".join(d for d in details if d) or None,
    )

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

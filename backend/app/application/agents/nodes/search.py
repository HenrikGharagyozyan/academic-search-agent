import logging

from app.application.agents.activity import ActivityRecorder, short_host
from app.application.agents.constants import MAX_SOURCES
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.domain.text.plain import to_label
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


def search_node(state: ResearchState, search_provider: SearchProvider) -> dict:
    query = state.get("search_query") or state["question"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    recorder.record("search", f"Searching the web for “{query}”")

    try:
        results = search_provider.search(query, limit=MAX_SOURCES)
    except Exception as exc:
        logger.error("Search failed for query=%r: %s", query, exc)
        raise UpstreamServiceError(f"Search provider failed: {exc}") from exc

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

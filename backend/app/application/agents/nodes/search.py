import logging

from app.application.agents.state import ResearchState
from app.application.agents.constants import MAX_SOURCES
from app.core.exceptions import UpstreamServiceError
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


def search_node(state: ResearchState, search_provider: SearchProvider) -> dict:
    query = state.get("search_query") or state["question"]

    try:
        results = search_provider.search(query, limit=MAX_SOURCES)
    except Exception as exc:
        logger.error("Firecrawl search failed for query=%r: %s", query, exc)
        raise UpstreamServiceError(f"Search provider failed: {exc}") from exc

    return {"search_results": results}
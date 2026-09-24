import logging

from app.agents.state import ResearchState
from app.agents.constants import MAX_SOURCES
from app.core.exceptions import UpstreamServiceError
from app.infrastructure.search.firecrawl import FirecrawlProvider

logger = logging.getLogger(__name__)


def search_node(state: ResearchState, firecrawl: FirecrawlProvider) -> dict:
    query = state.get("search_query") or state["question"]

    try:
        results = firecrawl.search(query, limit=MAX_SOURCES)
    except Exception as exc:
        logger.error("Firecrawl search failed for query=%r: %s", query, exc)
        raise UpstreamServiceError(f"Search provider failed: {exc}") from exc

    return {"search_results": results}
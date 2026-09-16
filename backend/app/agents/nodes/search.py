from app.agents.constants import MAX_SOURCES
from app.agents.state import ResearchState
from app.providers.firecrawl_provider import FirecrawlProvider


def search_node(
    state: ResearchState, firecrawl: FirecrawlProvider, limit: int = MAX_SOURCES
) -> dict:
    results = firecrawl.search(state["search_query"], limit=limit)
    return {"search_results": results}
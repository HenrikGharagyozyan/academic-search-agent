from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider


def refine_query_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    new_query = gemini.refine_query(state["question"], state["search_query"])
    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
    }
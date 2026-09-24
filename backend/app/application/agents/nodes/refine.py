import logging

from app.application.agents.state import ResearchState
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def refine_query_node(state: ResearchState, llm: LLMProvider) -> dict:
    try:
        new_query = llm.refine_query(state["question"], state["search_query"])
    except Exception:
        logger.warning("Failed to refine query, keeping previous query", exc_info=True)
        new_query = state["search_query"]

    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
    }

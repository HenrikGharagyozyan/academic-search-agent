import logging

from app.application.agents.activity import ActivityRecorder
from app.application.agents.state import ResearchState
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def refine_query_node(state: ResearchState, llm: LLMProvider) -> dict:
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    try:
        new_query = llm.refine_query(state["question"], state["search_query"])
    except Exception:
        logger.warning("Failed to refine query, keeping previous query", exc_info=True)
        new_query = state["search_query"]
        recorder.record("refine", "Could not rewrite the query, retrying as is")
    else:
        recorder.record(
            "refine",
            f"Not enough evidence — searching again for “{new_query}”",
        )

    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
        "activity": recorder.steps,
    }

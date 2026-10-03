import logging

from app.application.agents.activity import ActivityRecorder
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def refine_query_node(state: ResearchState, llm: LLMProvider) -> dict:
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    # What was searched and why it was not enough. With only the question and
    # its last query, the rewrite could do nothing but rephrase a search that
    # had already been run four ways.
    searched = state.get("search_queries") or [state["search_query"]]
    shortfall = state.get("shortfall", "")

    try:
        new_query = llm.refine_query(state["question"], searched, shortfall)
    except UpstreamServiceError:
        raise
    except Exception:
        logger.warning("Failed to refine query, keeping previous query", exc_info=True)
        new_query = state["search_query"]
        recorder.record("refine", "Could not rewrite the query, retrying as is")
    else:
        recorder.record(
            "refine",
            f"Not enough evidence — searching again for “{new_query}”",
            detail=shortfall or None,
        )

    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
        "activity": recorder.steps,
    }

import logging
from datetime import date

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import MAX_QUERIES, RECENCY_WINDOW_YEARS
from app.application.agents.state import ResearchState
from app.domain.query import has_recency_intent
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def plan_searches_node(state: ResearchState, llm: LLMProvider) -> dict:
    """Turns the question into several searches that pull in different directions.

    One query is why a well-studied topic comes back as a single paper: search
    ranks by links, so the most-cited work and its mirrors take every slot.
    """
    question = state.get("search_query") or state["question"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    recent = has_recency_intent(question)
    since_year = date.today().year - RECENCY_WINDOW_YEARS if recent else None

    try:
        plan = llm.plan_searches(question, count=MAX_QUERIES, recent=recent)
        queries = [q.strip() for q in plan.queries if q.strip()][:MAX_QUERIES]
    except Exception:
        logger.warning("Query planning failed, searching the question as-is", exc_info=True)
        queries = []

    # The question itself is always searched. If planning failed it is the only
    # query, which is exactly the behaviour this node replaced — degraded, but
    # never worse than before.
    if question not in queries:
        queries.insert(0, question)
    queries = queries[:MAX_QUERIES]

    recorder.record(
        "plan",
        f"Planned {count(len(queries), 'search')}"
        + (f", limited to {since_year} onwards" if since_year else ""),
        detail=" · ".join(queries[1:]) or None,
    )

    return {
        "search_queries": queries,
        "since_year": since_year,
        "activity": recorder.steps,
    }

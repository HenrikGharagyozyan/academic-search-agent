import logging

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import MAX_QUERIES
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.domain.query import anchor_terms, has_recency_intent, stays_on_topic
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def plan_searches_node(state: ResearchState, llm: LLMProvider) -> dict:
    """Turns the question into several searches that pull in different directions.

    One query is why a well-studied topic comes back as a single paper: search
    ranks by links, so the most-cited work and its mirrors take every slot.
    """
    question = state.get("search_query") or state["question"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    # Only steers the planner's wording. Results are not cut off by date: the
    # filter dropped good older work and the answer was thinner for it.
    recent = has_recency_intent(question)

    try:
        plan = llm.plan_searches(question, count=MAX_QUERIES, recent=recent)
        proposed = [q.strip() for q in plan.queries if q.strip()]
    except UpstreamServiceError:
        raise
    except Exception:
        logger.warning("Query planning failed, searching the question as-is", exc_info=True)
        proposed = []

    # The prompt asks the planner to stay in the subject; this enforces it. Asked
    # for different directions a model generalises, and a query that keeps none
    # of the question's terms searches a neighbouring field — which is how an
    # answer about multimode fibers came back about the Born approximation.
    anchors = anchor_terms(question)
    queries, drifted = [], []
    for query in proposed:
        (queries if stays_on_topic(query, anchors) else drifted).append(query)

    if drifted:
        logger.info("Discarded off-topic planned queries: %s", drifted)
        recorder.record(
            "plan",
            f"Discarded {count(len(drifted), 'query')} that left the subject",
            detail=" · ".join(drifted),
        )

    # The question itself is always searched. If planning failed it is the only
    # query, which is exactly the behaviour this node replaced — degraded, but
    # never worse than before.
    if question not in queries:
        queries.insert(0, question)
    queries = queries[:MAX_QUERIES]

    recorder.record(
        "plan",
        f"Planned {count(len(queries), 'search')}",
        detail=" · ".join(queries[1:]) or None,
    )

    return {
        "search_queries": queries,
        "activity": recorder.steps,
    }

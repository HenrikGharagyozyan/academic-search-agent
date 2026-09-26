import logging

from app.application.agents.state import ResearchState
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def grade_relevance_node(state: ResearchState, llm: LLMProvider) -> dict:
    selected = state["selected_chunks"]
    if not selected:
        return {"selected_chunks": []}

    try:
        grade = llm.grade_relevance(state["question"], selected)
    except Exception:
        logger.warning("Relevance grading failed, keeping all chunks", exc_info=True)
        return {}

    logger.info(
        "Relevance grade: %d/%d chunks kept. Reasoning: %s",
        len(grade.relevant_chunk_ids), len(selected), grade.reasoning,
    )

    relevant_ids = set(grade.relevant_chunk_ids)
    # An empty result is kept as-is: an answer built on chunks the judge
    # rejected is worse than no answer at all.
    return {"selected_chunks": [c for c in selected if c.chunk_id in relevant_ids]}


def grade_answer_node(state: ResearchState, llm: LLMProvider) -> dict:
    if not state["claims"]:
        return {"evidence_sufficient": False}

    try:
        grade = llm.grade_answer_quality(
            state["question"], state["summary"], state["claims"], state["conclusion"]
        )
    except Exception:
        logger.warning("Answer quality grading failed, trusting groundedness check", exc_info=True)
        return {}

    logger.info(
        "Answer quality grade: satisfactory=%s. Reasoning: %s",
        grade.is_satisfactory, grade.reasoning,
    )

    if not grade.is_satisfactory:
        return {"evidence_sufficient": False}

    return {}

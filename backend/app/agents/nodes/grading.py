import logging

from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


def grade_relevance_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["selected_chunks"]:
        return {"selected_chunks": []}

    chunks = [{"chunk_id": c.chunk_id, "text": c.text} for c in state["selected_chunks"]]

    try:
        grade = gemini.grade_relevance(state["question"], chunks)
    except Exception:
        logger.warning("Relevance grading failed, keeping all chunks", exc_info=True)
        return {}

    logger.info(
        "Relevance grade: %d/%d chunks kept. Reasoning: %s",
        len(grade.relevant_chunk_ids), len(chunks), grade.reasoning,
    )

    relevant_ids = set(grade.relevant_chunk_ids)
    filtered = [c for c in state["selected_chunks"] if c.chunk_id in relevant_ids]

    if not filtered:
        return {"selected_chunks": []}

    return {"selected_chunks": filtered}


def grade_answer_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["claims"]:
        return {"evidence_sufficient": False}

    try:
        grade = gemini.grade_answer_quality(
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
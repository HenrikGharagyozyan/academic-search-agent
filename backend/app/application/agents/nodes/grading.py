import logging

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.state import ResearchState
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def grade_relevance_node(state: ResearchState, llm: LLMProvider) -> dict:
    selected = state["selected_chunks"]
    if not selected:
        return {"selected_chunks": []}

    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    try:
        grade = llm.grade_relevance(state["question"], selected)
    except Exception:
        logger.warning("Relevance grading failed, keeping all chunks", exc_info=True)
        recorder.record(
            "grade_relevance",
            "Could not judge passage relevance, keeping all of them",
            detail=f"{count(len(selected), 'passage')} kept",
        )
        return {"activity": recorder.steps}

    logger.info(
        "Relevance grade: %d/%d chunks kept. Reasoning: %s",
        len(grade.relevant_chunk_ids), len(selected), grade.reasoning,
    )

    relevant_ids = set(grade.relevant_chunk_ids)
    kept = [c for c in selected if c.chunk_id in relevant_ids]

    recorder.record(
        "grade_relevance",
        f"Judged {count(len(selected), 'passage')}, {len(kept)} on topic",
        detail=grade.reasoning or None,
    )

    # An empty result is kept as-is: an answer built on chunks the judge
    # rejected is worse than no answer at all.
    return {"selected_chunks": kept, "activity": recorder.steps}


def grade_answer_node(state: ResearchState, llm: LLMProvider) -> dict:
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    if not state["claims"]:
        recorder.record("grade_answer", "No grounded claims to review")
        return {"evidence_sufficient": False, "activity": recorder.steps}

    try:
        grade = llm.grade_answer_quality(
            state["question"], state["summary"], state["claims"], state["conclusion"]
        )
    except Exception:
        logger.warning("Answer quality grading failed, trusting groundedness check", exc_info=True)
        recorder.record(
            "grade_answer",
            "Could not review the answer, trusting the groundedness check",
        )
        return {"activity": recorder.steps}

    logger.info(
        "Answer quality grade: satisfactory=%s. Reasoning: %s",
        grade.is_satisfactory, grade.reasoning,
    )

    recorder.record(
        "grade_answer",
        "Reviewed the answer: satisfactory" if grade.is_satisfactory
        else "Reviewed the answer: not good enough",
        detail=grade.reasoning or None,
    )

    if not grade.is_satisfactory:
        return {"evidence_sufficient": False, "activity": recorder.steps}

    return {"activity": recorder.steps}

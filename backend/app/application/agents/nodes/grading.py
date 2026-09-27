import logging
from concurrent.futures import ThreadPoolExecutor

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import GRADE_BATCH_SIZE
from app.application.agents.state import ResearchState
from app.domain.documents import Chunk
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def _judge_batch(
    llm: LLMProvider, question: str, batch: list[Chunk]
) -> tuple[list[str], str] | None:
    """The ids this batch's judge kept, or None if the call failed."""
    try:
        grade = llm.grade_relevance(question, batch)
    except Exception:
        logger.warning("Relevance grading failed for a batch", exc_info=True)
        return None
    return list(grade.relevant_chunk_ids), grade.reasoning


def grade_relevance_node(state: ResearchState, llm: LLMProvider) -> dict:
    selected = state["selected_chunks"]
    if not selected:
        return {"selected_chunks": []}

    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    # Judged in batches, not in one call. A single call over forty passages kept
    # under a fifth of them; batches of eight keep about half. The judge cannot
    # weigh forty heterogeneous passages at once, and the prompt is not the
    # lever — rewriting it moved the count by one.
    batches = [
        selected[i : i + GRADE_BATCH_SIZE]
        for i in range(0, len(selected), GRADE_BATCH_SIZE)
    ]

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        outcomes = list(
            executor.map(lambda b: _judge_batch(llm, state["question"], b), batches)
        )

    relevant_ids: set[str] = set()
    reasons: list[str] = []
    failed = 0

    for batch, outcome in zip(batches, outcomes):
        if outcome is None:
            # A batch whose judge could not be reached keeps its passages, so one
            # failed call costs nothing rather than discarding what it held.
            failed += 1
            relevant_ids.update(c.chunk_id for c in batch)
            continue
        ids, reasoning = outcome
        relevant_ids.update(ids)
        if reasoning:
            reasons.append(reasoning)

    kept = [c for c in selected if c.chunk_id in relevant_ids]

    logger.info(
        "Relevance grade: %d/%d chunks kept across %d batches (%d failed)",
        len(kept), len(selected), len(batches), failed,
    )

    detail = " · ".join(reasons) or None
    if failed:
        detail = f"{count(failed, 'batch', 'batches')} could not be judged; " + (detail or "")

    recorder.record(
        "grade_relevance",
        f"Judged {count(len(selected), 'passage')} in "
        f"{count(len(batches), 'batch', 'batches')}, {len(kept)} on topic",
        detail=detail,
    )

    # An empty result is kept as-is: an answer built on passages the judge
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

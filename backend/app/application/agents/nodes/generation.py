import logging

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.state import ResearchState
from app.domain.answers import ClaimsResponse
from app.domain.text.cleanup import find_evidence_id_leak, strip_evidence_ids
from app.domain.text.latex import restore_latex
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)

# Generations to try before giving up on one that keeps leaking evidence ids.
# The stripping catches the forms seen so far; a second leak in a row means a
# form it has never seen, and a fresh generation rarely repeats it.
MAX_GENERATION_ATTEMPTS = 2


def _presentable(text: str) -> str:
    """Turns raw model text into what the reader should actually see."""
    return strip_evidence_ids(restore_latex(text))


def _presented(result: ClaimsResponse) -> ClaimsResponse:
    return result.model_copy(
        update={
            "summary": _presentable(result.summary),
            "claims": [
                claim.model_copy(update={"text": _presentable(claim.text)})
                for claim in result.claims
            ],
            "conclusion": _presentable(result.conclusion),
        }
    )


def _leak(result: ClaimsResponse) -> str | None:
    """An evidence id the stripping missed, anywhere the reader would see it."""
    texts = [result.summary, result.conclusion]
    texts += [t for claim in result.claims for t in (claim.text, claim.theme)]
    return next((leak for t in texts if (leak := find_evidence_id_leak(t))), None)


def generate_claims_node(state: ResearchState, llm: LLMProvider) -> dict:
    empty = {"summary": "", "claims": [], "conclusion": ""}
    if not state["selected_chunks"]:
        return empty

    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            result = _presented(
                llm.generate_answer(state["question"], state["selected_chunks"])
            )
        except Exception:
            logger.warning("Failed to generate answer, returning empty", exc_info=True)
            recorder.record("generate", "Could not write an answer from the passages")
            return empty | {"activity": recorder.steps}

        leak = _leak(result)
        if leak is None:
            break
        # Internal ids are not for the reader, and the stripping has just
        # failed to remove one. That is a bug in the stripping, so it is logged
        # loudly, and the answer is not handed on with the id in it.
        logger.warning(
            "Evidence id survived cleanup (attempt %d/%d): %r",
            attempt, MAX_GENERATION_ATTEMPTS, leak,
        )
    else:
        recorder.record(
            "generate",
            "Could not write an answer without internal ids in it",
            detail=f"{count(MAX_GENERATION_ATTEMPTS, 'attempt')} leaked evidence ids",
        )
        return empty | {"activity": recorder.steps}

    recorder.record(
        "generate",
        f"Wrote {count(len(result.claims), 'claim')} from {count(len(state['selected_chunks']), 'passage')}",
    )

    return {
        "summary": result.summary,
        "claims": result.claims,
        "conclusion": result.conclusion,
        "activity": recorder.steps,
    }

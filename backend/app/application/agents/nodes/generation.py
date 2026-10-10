import logging
from dataclasses import dataclass

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.state import ResearchState
from app.core.exceptions import UpstreamServiceError
from app.domain.answers import ClaimsResponse
from app.domain.attribution import move_attribution_out_of_theme
from app.domain.text.cleanup import find_evidence_id_leak, strip_evidence_ids
from app.domain.text.equations import (
    display_equations,
    plain_text_equations,
    states_an_equation,
)
from app.domain.text.headings import tidy_theme
from app.domain.text.latex import has_damaged_maths, restore_latex
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)

# Generations to try before accepting a flawed one. A flaw is something the
# filters could not put right — an evidence id in a form they have never seen,
# a formula JSON mangled beyond repair — and a fresh generation rarely repeats it.
MAX_GENERATION_ATTEMPTS = 2


def _presentable(text: str) -> str:
    """Turns raw model text into what the reader should actually see."""
    return strip_evidence_ids(restore_latex(text))


def _presented(result: ClaimsResponse) -> ClaimsResponse:
    # A direct answer has no sections. The model is told so; this holds it to
    # its own decision, so "what is X" cannot arrive as a numbered survey.
    themed = result.answer_shape == "survey"
    claims = []
    for claim in result.claims:
        # Before the theme can be dropped: an attribution written only in the
        # heading would otherwise go with it.
        theme, text = move_attribution_out_of_theme(claim.theme, claim.text)
        claims.append(
            claim.model_copy(
                update={
                    "text": _presentable(text),
                    "theme": tidy_theme(theme) if themed else "",
                }
            )
        )
    return result.model_copy(
        update={
            "summary": _presentable(result.summary),
            "claims": claims,
            "conclusion": _presentable(result.conclusion),
        }
    )


def _texts(result: ClaimsResponse) -> list[str]:
    """Everything in the answer a reader would see."""
    texts = [result.summary, result.conclusion]
    return texts + [t for claim in result.claims for t in (claim.text, claim.theme)]


def _leak(result: ClaimsResponse) -> str | None:
    """An evidence id the stripping missed, anywhere the reader would see it."""
    return next((leak for t in _texts(result) if (leak := find_evidence_id_leak(t))), None)


@dataclass
class _Attempt:
    """One generation, and what is still wrong with it after the filters."""

    result: ClaimsResponse
    # An evidence id the stripping missed. The one flaw an answer is withheld for.
    leak: str | None
    # A formula holding a control character that cannot be traced to a command.
    damaged: bool
    # Equations written as ordinary characters rather than set as mathematics.
    plain: list[str]
    # The passages state equations and the answer states none.
    omitted: bool

    @property
    def flaws(self) -> int:
        return bool(self.leak) * 100 + self.damaged * 10 + self.omitted * 5 + len(self.plain)


def _examine(raw: ClaimsResponse, stated: int) -> _Attempt:
    result = _presented(raw)
    return _Attempt(
        result=result,
        leak=_leak(result),
        # Judged on the raw text: once the control characters are stripped, a
        # formula that lost its Ω is indistinguishable from one that never had it.
        damaged=any(has_damaged_maths(t) for t in _texts(raw)),
        plain=[eq for t in _texts(result) for eq in plain_text_equations(t)],
        # Deliberately coarse. Which of the passages' equations is the one that
        # matters is a judgement; an answer with no equation at all, written
        # from passages that set several, is not.
        omitted=stated > 0 and not any(states_an_equation(t) for t in _texts(result)),
    )


def generate_claims_node(state: ResearchState, llm: LLMProvider) -> dict:
    empty = {"summary": "", "claims": [], "conclusion": ""}
    if not state["selected_chunks"]:
        return empty

    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))
    attempts: list[_Attempt] = []
    # Equations a passage sets on a line of their own, as a page sets the ones
    # it is about. They are facts of the evidence like any other.
    stated = sum(len(display_equations(c.text)) for c in state["selected_chunks"])

    for number in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            raw = llm.generate_answer(state["question"], state["selected_chunks"])
        except UpstreamServiceError:
            raise
        except Exception:
            logger.warning("Failed to generate answer, returning empty", exc_info=True)
            if attempts:
                # A retry that fails does not cost the answer already in hand.
                break
            recorder.record("generate", "Could not write an answer from the passages")
            return empty | {"activity": recorder.steps}

        attempt = _examine(raw, stated)
        attempts.append(attempt)
        if not attempt.flaws:
            break

        # Each of these is something the prompt forbids and the filters could
        # not put right, so it is logged loudly rather than passed over.
        if attempt.leak is not None:
            logger.warning(
                "Evidence id survived cleanup (attempt %d/%d): %r",
                number, MAX_GENERATION_ATTEMPTS, attempt.leak,
            )
        if attempt.damaged:
            logger.warning(
                "A formula arrived damaged beyond repair (attempt %d/%d): the model "
                "wrote LaTeX with backslashes and JSON decoded them",
                number, MAX_GENERATION_ATTEMPTS,
            )
        if attempt.omitted:
            logger.warning(
                "The passages state %d equation(s) and the answer states none "
                "(attempt %d/%d)",
                stated, number, MAX_GENERATION_ATTEMPTS,
            )
        if attempt.plain:
            logger.warning(
                "Equations left as plain text (attempt %d/%d): %s",
                number, MAX_GENERATION_ATTEMPTS, attempt.plain,
            )

    # The retry is not always the better one, so the least flawed is kept.
    best = min(attempts, key=lambda a: a.flaws)

    if best.leak is not None:
        # Internal ids are not for the reader: an answer that still carries one
        # is not handed on.
        recorder.record(
            "generate",
            "Could not write an answer without internal ids in it",
            detail=f"{count(len(attempts), 'attempt')} leaked evidence ids",
        )
        return empty | {"activity": recorder.steps}

    result = best.result
    detail = f"answered as a {result.answer_shape} question"
    if best.damaged:
        # Not withheld: the prose is sound and one symbol is not worth the
        # answer. But the reader is told, because a formula missing a symbol
        # reads as a different formula.
        detail += "; a formula may be missing a symbol the model's output lost"
    if best.omitted:
        # Said out loud: a formula the sources give and the answer lacks must
        # not go missing silently.
        detail += (
            f"; the passages state {count(stated, 'equation')} "
            "and the answer gives none"
        )

    recorder.record(
        "generate",
        f"Wrote {count(len(result.claims), 'claim')} from {count(len(state['selected_chunks']), 'passage')}",
        detail=detail,
    )

    return {
        "summary": result.summary,
        "claims": result.claims,
        "conclusion": result.conclusion,
        "activity": recorder.steps,
    }

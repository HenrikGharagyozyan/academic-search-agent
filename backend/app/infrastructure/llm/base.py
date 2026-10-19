"""Everything a LangChain-backed provider does, written once.

A vendor adapter supplies a chat model and two names. Prompt assembly,
structured output and the retry policy live here, so adding a provider cannot
quietly change how the pipeline behaves.
"""

import functools
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeVar

from langchain_core.language_models import BaseChatModel

from app.core.exceptions import ProviderCreditsExhausted
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import AnswerQualityGrade, RelevanceGrade
from app.domain.query import QueryPlan
from app.domain.text.latex import to_channel_notation
from app.infrastructure.llm.prompts import (
    ANSWER_PROMPT,
    ANSWER_QUALITY_PROMPT,
    EXPANSION_PROMPT,
    NO_RECENCY_INSTRUCTION,
    RECENCY_INSTRUCTION,
    REFINE_PROMPT,
    RELEVANCE_GRADE_PROMPT,
)
from app.infrastructure.llm.retry import is_billing_error, llm_retry

P = ParamSpec("P")
R = TypeVar("R")


def _billing_surfaces(method: Callable[P, R]) -> Callable[P, R]:
    """Turns a refusal to pay into an error the reader is shown.

    The nodes treat a failed model call as a degraded step and carry on, which
    is right for an outage in one call and wrong for an empty balance: every
    call of the run fails, and the answer arrives as "no reliable sources".
    """

    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if not is_billing_error(exc):
                raise
            provider = getattr(args[0], "provider_name", "the language model provider")
            raise ProviderCreditsExhausted(
                f"The language model provider ({provider}) is out of credits and "
                "refused the request. Top up the account and try again."
            ) from exc

    return wrapper


def _structured(runnable, prompt):
    """Invokes a structured-output runnable, refusing an empty reply.

    Over tool calls a reply that never calls the tool — cut off by the token
    ceiling, or answered in prose — parses to None, and the nodes then fail on
    an attribute of None. Raised instead, it is retried like any transient
    failure.
    """
    result = runnable.invoke(prompt)
    if result is None:
        raise RuntimeError("The model returned no structured output")
    return result


class LangChainLLMProvider:
    """Base adapter for any model reachable through a LangChain chat interface."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        provider_name: str,
        model_name: str,
        short_llm: BaseChatModel | None = None,
        structured_method: str | None = None,
    ) -> None:
        """``short_llm`` is the same model configured for the calls whose reply
        is a few hundred tokens — planning, grading, rewriting a query — when
        the vendor charges for the response ceiling rather than the response.
        Only the answer itself needs a long one.

        ``structured_method`` is passed to ``with_structured_output`` for a
        model that offers tool calls but not JSON-schema output."""
        short_llm = short_llm or llm
        structured = {"method": structured_method} if structured_method else {}
        self._llm = short_llm
        self._provider_name = provider_name
        self._model_name = model_name
        self._answer_llm = llm.with_structured_output(ClaimsResponse, **structured)
        self._plan_llm = short_llm.with_structured_output(QueryPlan, **structured)
        self._relevance_llm = short_llm.with_structured_output(RelevanceGrade, **structured)
        self._quality_llm = short_llm.with_structured_output(AnswerQualityGrade, **structured)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _as_block(chunks: Sequence[Chunk], label: str) -> str:
        return "\n\n".join(f"[{label}: {c.chunk_id}]\n{c.text}" for c in chunks)

    @staticmethod
    def _as_evidence_block(chunks: Sequence[Chunk]) -> str:
        return LangChainLLMProvider._as_block(chunks, "evidence_id")

    @classmethod
    def _as_evidence(cls, chunks: Sequence[Chunk]) -> str:
        # In @ notation, because the model copies the notation it reads: shown
        # backslashes, it wrote backslashes, and JSON ate them.
        return to_channel_notation(cls._as_evidence_block(chunks))

    @_billing_surfaces
    @llm_retry
    def plan_searches(self, question: str, count: int, recent: bool) -> QueryPlan:
        prompt = EXPANSION_PROMPT.invoke(
            {
                "question": question,
                "query_count": count,
                "recency_instruction": (
                    RECENCY_INSTRUCTION if recent else NO_RECENCY_INSTRUCTION
                ),
            }
        )
        return _structured(self._plan_llm, prompt)

    @_billing_surfaces
    @llm_retry
    def generate_answer(self, question: str, evidence: Sequence[Chunk]) -> ClaimsResponse:
        prompt = ANSWER_PROMPT.invoke(
            {"question": question, "evidence_block": self._as_evidence(evidence)}
        )
        return _structured(self._answer_llm, prompt)

    @_billing_surfaces
    @llm_retry
    def refine_query(
        self, question: str, previous_queries: Sequence[str], shortfall: str
    ) -> str:
        prompt = REFINE_PROMPT.invoke(
            {
                "question": question,
                "previous_queries": "\n".join(f"- {q}" for q in previous_queries),
                "shortfall": shortfall or "not recorded",
            }
        )
        # `.text` over `.content`: content is a string for some models and a list
        # of content blocks for others, and only `.text` flattens both.
        return self._llm.invoke(prompt).text.strip()

    @_billing_surfaces
    @llm_retry
    def grade_relevance(self, question: str, chunks: Sequence[Chunk]) -> RelevanceGrade:
        prompt = RELEVANCE_GRADE_PROMPT.invoke(
            {"question": question, "chunks_block": self._as_block(chunks, "chunk_id")}
        )
        return _structured(self._relevance_llm, prompt)

    @_billing_surfaces
    @llm_retry
    def grade_answer_quality(
        self, question: str, summary: str, claims: Sequence[Claim], conclusion: str
    ) -> AnswerQualityGrade:
        prompt = ANSWER_QUALITY_PROMPT.invoke(
            {
                "question": question,
                "summary": summary,
                "claims_block": "\n".join(f"- {c.text}" for c in claims),
                "conclusion": conclusion,
            }
        )
        return _structured(self._quality_llm, prompt)

"""Everything a LangChain-backed provider does, written once.

A vendor adapter supplies a chat model and two names. Prompt assembly,
structured output and the retry policy live here, so adding a provider cannot
quietly change how the pipeline behaves.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import AnswerQualityGrade, RelevanceGrade
from app.infrastructure.llm.prompts import (
    ANSWER_PROMPT,
    ANSWER_QUALITY_PROMPT,
    REFINE_PROMPT,
    RELEVANCE_GRADE_PROMPT,
)
from app.infrastructure.llm.retry import llm_retry


class LangChainLLMProvider:
    """Base adapter for any model reachable through a LangChain chat interface."""

    def __init__(self, llm: BaseChatModel, *, provider_name: str, model_name: str) -> None:
        self._llm = llm
        self._provider_name = provider_name
        self._model_name = model_name
        self._answer_llm = llm.with_structured_output(ClaimsResponse)
        self._relevance_llm = llm.with_structured_output(RelevanceGrade)
        self._quality_llm = llm.with_structured_output(AnswerQualityGrade)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _as_block(chunks: Sequence[Chunk], label: str) -> str:
        return "\n\n".join(f"[{label}: {c.chunk_id}]\n{c.text}" for c in chunks)

    @llm_retry
    def generate_answer(self, question: str, evidence: Sequence[Chunk]) -> ClaimsResponse:
        prompt = ANSWER_PROMPT.invoke(
            {"question": question, "evidence_block": self._as_block(evidence, "evidence_id")}
        )
        return self._answer_llm.invoke(prompt)

    @llm_retry
    def refine_query(self, question: str, previous_query: str) -> str:
        prompt = REFINE_PROMPT.invoke(
            {"question": question, "previous_query": previous_query}
        )
        # `.text` over `.content`: content is a string for some models and a list
        # of content blocks for others, and only `.text` flattens both.
        return self._llm.invoke(prompt).text.strip()

    @llm_retry
    def grade_relevance(self, question: str, chunks: Sequence[Chunk]) -> RelevanceGrade:
        prompt = RELEVANCE_GRADE_PROMPT.invoke(
            {"question": question, "chunks_block": self._as_block(chunks, "chunk_id")}
        )
        return self._relevance_llm.invoke(prompt)

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
        return self._quality_llm.invoke(prompt)

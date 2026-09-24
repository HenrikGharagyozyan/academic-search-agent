from collections.abc import Sequence

from app.core.config import get_settings
from app.infrastructure.llm.retry import llm_retry
from app.infrastructure.llm.prompts import (
    ANSWER_PROMPT, REFINE_PROMPT, ANSWER_QUALITY_PROMPT, RELEVANCE_GRADE_PROMPT,
)
from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import RelevanceGrade, AnswerQualityGrade
from langchain_openai import ChatOpenAI

REQUEST_TIMEOUT_SECONDS = 30


class OpenRouterProvider:
    """Drop-in replacement for GeminiProvider for grading/generation.
    Embeddings still use Gemini and are not part of this class."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._structured_llm = self._llm.with_structured_output(ClaimsResponse)

    @llm_retry
    def generate_answer(self, question: str, evidence: Sequence[Chunk]) -> ClaimsResponse:
        evidence_block = "\n\n".join(
            f"[evidence_id: {c.chunk_id}]\n{c.text}" for c in evidence
        )
        prompt_value = ANSWER_PROMPT.invoke(
            {"question": question, "evidence_block": evidence_block}
        )
        result: ClaimsResponse = self._structured_llm.invoke(prompt_value)
        return result

    @llm_retry
    def refine_query(self, question: str, previous_query: str) -> str:
        prompt_value = REFINE_PROMPT.invoke(
            {"question": question, "previous_query": previous_query}
        )
        response = self._llm.invoke(prompt_value)
        return response.text.strip()

    @llm_retry
    def grade_relevance(self, question: str, chunks: Sequence[Chunk]) -> RelevanceGrade:
        chunks_block = "\n\n".join(
            f"[chunk_id: {c.chunk_id}]\n{c.text}" for c in chunks
        )
        prompt_value = RELEVANCE_GRADE_PROMPT.invoke(
            {"question": question, "chunks_block": chunks_block}
        )
        structured = self._llm.with_structured_output(RelevanceGrade)
        return structured.invoke(prompt_value)

    @llm_retry
    def grade_answer_quality(
        self, question: str, summary: str, claims: Sequence[Claim], conclusion: str
    ) -> AnswerQualityGrade:
        claims_block = "\n".join(f"- {c.text}" for c in claims)
        prompt_value = ANSWER_QUALITY_PROMPT.invoke(
            {
                "question": question,
                "summary": summary,
                "claims_block": claims_block,
                "conclusion": conclusion,
            }
        )
        structured = self._llm.with_structured_output(AnswerQualityGrade)
        return structured.invoke(prompt_value)
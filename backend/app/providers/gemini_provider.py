from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.providers.gemini_prompts import ANSWER_PROMPT, REFINE_PROMPT, ANSWER_QUALITY_PROMPT, RELEVANCE_GRADE_PROMPT
from app.schemas.answer import Claim, ClaimsResponse
from app.schemas.grading import RelevanceGrade, AnswerQualityGrade
from langchain_google_genai import ChatGoogleGenerativeAI


def _is_transient_error(exc: BaseException) -> bool:
    message = str(exc)
    return "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message or "UNAVAILABLE" in message


gemini_retry = retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=settings.gemini_api_key,
        )
        self._structured_llm = self._llm.with_structured_output(ClaimsResponse)

    @gemini_retry
    def generate_answer(
        self, question: str, evidence_chunks: list[dict]
    ) -> ClaimsResponse:
        evidence_block = "\n\n".join(
            f"[evidence_id: {c['chunk_id']}]\n{c['text']}" for c in evidence_chunks
        )

        prompt_value = ANSWER_PROMPT.invoke(
            {"question": question, "evidence_block": evidence_block}
        )
        result: ClaimsResponse = self._structured_llm.invoke(prompt_value)

        return result

    @gemini_retry
    def refine_query(self, question: str, previous_query: str) -> str:
        prompt_value = REFINE_PROMPT.invoke(
            {"question": question, "previous_query": previous_query}
        )
        response = self._llm.invoke(prompt_value)
        return response.content.strip()

    @gemini_retry
    def grade_relevance(
        self, question: str, chunks: list[dict]
    ) -> RelevanceGrade:
        chunks_block = "\n\n".join(
            f"[chunk_id: {c['chunk_id']}]\n{c['text']}" for c in chunks
        )
        prompt_value = RELEVANCE_GRADE_PROMPT.invoke(
            {"question": question, "chunks_block": chunks_block}
        )
        structured = self._llm.with_structured_output(RelevanceGrade)
        return structured.invoke(prompt_value)

    @gemini_retry
    def grade_answer_quality(
        self, question: str, summary: str, claims: list[Claim], conclusion: str
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
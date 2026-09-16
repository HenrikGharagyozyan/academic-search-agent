from app.core.config import get_settings
from app.providers.gemini_prompts import ANSWER_PROMPT, REFINE_PROMPT
from app.schemas.answer import Claim, ClaimsResponse
from langchain_google_genai import ChatGoogleGenerativeAI



class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=settings.gemini_api_key,
        )
        self._structured_llm = self._llm.with_structured_output(ClaimsResponse)

    def generate_claims(
        self, question: str, evidence_chunks: list[dict]
    ) -> list[Claim]:
        evidence_block = "\n\n".join(
            f"[evidence_id: {c['chunk_id']}]\n{c['text']}" for c in evidence_chunks
        )

        # chain = PROMPT | self._structured_llm
        # result: ClaimsResponse = chain.invoke(
        #     {"question": question, "evidence_block": evidence_block}
        # )
        
        # For test
        prompt_value = ANSWER_PROMPT.invoke(
            {"question": question, "evidence_block": evidence_block}
        )
        result: ClaimsResponse = self._structured_llm.invoke(prompt_value)

        return result.claims

    def refine_query(self, question: str, previous_query: str) -> str:
        prompt_value = REFINE_PROMPT.invoke(
            {"question": question, "previous_query": previous_query}
        )
        response = self._llm.invoke(prompt_value)
        return response.content.strip()
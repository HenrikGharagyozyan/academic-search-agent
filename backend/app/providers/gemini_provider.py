from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.schemas.answer import Claim, ClaimsResponse

SYSTEM_PROMPT = """You are a research assistant that answers questions strictly \
based on the provided evidence chunks. Rules:

1. Only use information present in the evidence. Never invent facts.
2. Every claim you make MUST cite at least one evidence_id from the provided list.
3. If the evidence does not answer the question, say so explicitly in a claim \
   with an empty evidence_ids list and confidence "low".
4. Never invent an evidence_id that was not provided.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nEvidence:\n{evidence_block}"),
    ]
)


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=settings.gemini_api_key,
        )
        self._structured_llm = llm.with_structured_output(ClaimsResponse)

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
        prompt_value = PROMPT.invoke(
            {"question": question, "evidence_block": evidence_block}
        )
        result: ClaimsResponse = self._structured_llm.invoke(prompt_value)

        return result.claims
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings
from app.infrastructure.llm.base import LangChainLLMProvider


class GeminiProvider(LangChainLLMProvider):
    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
                timeout=settings.llm_timeout_seconds,
            ),
            provider_name="gemini",
            model_name=model,
        )

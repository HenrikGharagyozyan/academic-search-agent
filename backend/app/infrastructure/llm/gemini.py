from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings
from app.infrastructure.llm.base import LangChainLLMProvider

REQUEST_TIMEOUT_SECONDS = 30


class GeminiProvider(LangChainLLMProvider):
    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ),
            provider_name="gemini",
            model_name=model,
        )

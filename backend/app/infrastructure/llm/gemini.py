from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings, get_settings
from app.infrastructure.llm.base import LangChainLLMProvider

DEFAULT_MODEL = "gemini-3.6-flash"
REQUEST_TIMEOUT_SECONDS = 30


class GeminiProvider(LangChainLLMProvider):
    def __init__(self, settings: Settings | None = None, model: str = DEFAULT_MODEL) -> None:
        settings = settings or get_settings()
        super().__init__(
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ),
            provider_name="gemini",
            model_name=model,
        )

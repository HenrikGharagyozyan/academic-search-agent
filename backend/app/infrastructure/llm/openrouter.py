from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.infrastructure.llm.base import LangChainLLMProvider

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 30


class OpenRouterProvider(LangChainLLMProvider):
    def __init__(self, settings: Settings | None = None, model: str = DEFAULT_MODEL) -> None:
        settings = settings or get_settings()
        super().__init__(
            ChatOpenAI(
                base_url=BASE_URL,
                api_key=settings.openrouter_api_key,
                model=model,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ),
            provider_name="openrouter",
            model_name=model,
        )

from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.infrastructure.llm.base import LangChainLLMProvider

BASE_URL = "https://openrouter.ai/api/v1"

# Ceiling for the calls that return a query plan, a grade or a rewritten query.
# OpenRouter reserves credit for the ceiling of every request in flight, and the
# relevance judge runs five batches at once: at the answer's ceiling that was
# five times 8192 tokens reserved for replies of a few hundred, and three of the
# five were refused on a balance that covered the real cost many times over.
SHORT_RESPONSE_MAX_TOKENS = 1024


class OpenRouterProvider(LangChainLLMProvider):
    def __init__(self, settings: Settings, model: str) -> None:
        def chat(max_tokens: int, reasoning: bool = False) -> ChatOpenAI:
            return ChatOpenAI(
                base_url=BASE_URL,
                api_key=settings.openrouter_api_key,
                model=model,
                max_tokens=max_tokens,
                timeout=settings.llm_timeout_seconds,
                # Said either way: some models reason unless told not to, and
                # Nemotron's reasoning alone overran the short calls' ceiling,
                # leaving no room for the tool call that carries the reply.
                extra_body={"reasoning": {"enabled": reasoning}},
            )

        super().__init__(
            chat(settings.llm_max_tokens, reasoning=settings.llm_reasoning),
            short_llm=chat(min(SHORT_RESPONSE_MAX_TOKENS, settings.llm_max_tokens)),
            provider_name="openrouter",
            model_name=model,
            # Tool calls are the one structured-output route every OpenRouter
            # model offers; free endpoints often lack JSON-schema output.
            structured_method="function_calling",
        )

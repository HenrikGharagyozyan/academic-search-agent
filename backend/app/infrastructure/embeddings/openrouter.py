from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.infrastructure.embeddings.gemini import GeminiEmbeddingsProvider
from app.infrastructure.llm.openrouter import BASE_URL


class OpenRouterEmbeddingsProvider(GeminiEmbeddingsProvider):
    """The same cache, batching and retry as the Gemini adapter, served through
    OpenRouter — whose key has no daily quota for the free tier to run out of."""

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self._embeddings = OpenAIEmbeddings(
            base_url=BASE_URL,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_embedding_model,
            # Plain strings, not tiktoken ids: OpenRouter routes to models whose
            # tokenizer is not OpenAI's.
            check_embedding_ctx_length=False,
        )

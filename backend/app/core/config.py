from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keys of the provider registry. Declared here so a misspelled LLM_PROVIDER is
# rejected when settings load, rather than falling back to another vendor and
# quietly spending the wrong quota.
ProviderName = Literal["gemini", "openrouter"]


class Settings(BaseSettings):
    firecrawl_api_key: str
    gemini_api_key: str
    openrouter_api_key: str | None = None

    llm_provider: ProviderName = "gemini"
    # Empty means "whatever the registry lists as this provider's default".
    llm_model: str = ""
    # Ceiling on each response. OpenRouter reserves credit for the ceiling before
    # the request runs, and with none set it reserves the model's maximum — 16384
    # tokens for gpt-4o-mini — so every call was refused with 402 on a balance
    # that would have covered the few thousand tokens an answer actually takes.
    llm_max_tokens: int = Field(default=8192, gt=0)
    # Seconds before a model call is abandoned. A reasoning model thinks before
    # it answers, and a long answer can take minutes.
    llm_timeout_seconds: float = Field(default=30, gt=0)
    # Ask an OpenRouter model to reason before writing the answer. Only the
    # answer call reasons; planning and grading stay quick.
    llm_reasoning: bool = False

    embedding_model: str = "models/gemini-embedding-001"
    # "openrouter" embeds through the OpenRouter key instead of Gemini's, whose
    # free quota covers about one question a day.
    embedding_provider: Literal["gemini", "openrouter"] = "gemini"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _require_selected_provider_key(self) -> "Settings":
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

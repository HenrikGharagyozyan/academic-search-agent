from functools import lru_cache
from typing import Literal

from pydantic import model_validator
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

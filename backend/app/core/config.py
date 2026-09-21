from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    firecrawl_api_key: str
    gemini_api_key: str
    openrouter_api_key: str | None = None
    llm_provider: Literal["gemini", "openrouter"] = "gemini"
    openrouter_model: str = "openai/gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _require_openrouter_key(self) -> "Settings":
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
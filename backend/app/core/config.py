from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    firecrawl_api_key: str
    gemini_api_key: str
    openrouter_api_key: str | None = None
    llm_provider: str = "gemini"  # "gemini" | "openrouter"
    openrouter_model: str = "openai/gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
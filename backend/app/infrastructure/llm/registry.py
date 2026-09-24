"""The one place that knows which language model providers exist.

Adding a vendor means adding its adapter and one entry here — nothing in the
agent, the services or the API changes, because all of them depend on the
LLMProvider port rather than on a class.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import ProviderName, Settings, get_settings
from app.infrastructure.llm.gemini import GeminiProvider
from app.infrastructure.llm.openrouter import OpenRouterProvider
from app.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    default_model: str
    build: Callable[[Settings, str], LLMProvider]


REGISTRY: dict[ProviderName, ProviderSpec] = {
    "gemini": ProviderSpec(default_model="gemini-3.6-flash", build=GeminiProvider),
    "openrouter": ProviderSpec(default_model="openai/gpt-4o-mini", build=OpenRouterProvider),
}


def available_providers() -> list[str]:
    return sorted(REGISTRY)


def resolve_model(settings: Settings) -> str:
    """The model to run: the configured override, else the provider's default."""
    return settings.llm_model or REGISTRY[settings.llm_provider].default_model


def create_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    spec = REGISTRY[settings.llm_provider]
    model = resolve_model(settings)

    logger.info("Using LLM provider=%s model=%s", settings.llm_provider, model)
    return spec.build(settings, model)

# tests/conftest.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_research_service
from app.core.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    # Environment variables take precedence over backend/.env, so tests never pick up
    # real keys: local runs behave exactly like CI, and nothing can hit a paid API.
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_research_service():
    return MagicMock()


@pytest.fixture
def client(mock_research_service):
    # The lifespan would otherwise build a real ResearchService (Firecrawl, Gemini,
    # Chroma): it needs API keys and every request would hit the real network.
    with patch("app.main.ResearchService", return_value=mock_research_service):
        app.dependency_overrides[get_research_service] = lambda: mock_research_service
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.clear()

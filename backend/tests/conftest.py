from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_document_service, get_research_service, get_search_service
from app.core.config import get_settings
from app.main import app
from app.schemas.grading import RelevanceGrade


@pytest.fixture
def keep_all_chunks_relevant():
    """side_effect для gemini.grade_relevance: судья признаёт релевантными все чанки.

    Тесты пайплайна проверяют не отбор чанков, а то, что происходит дальше, —
    поэтому судью нужно замокать явно. Иначе MagicMock отдаёт пустой
    relevant_chunk_ids, узел отбрасывает всё, и до генерации ничего не доходит.
    """

    def _grade(question, chunks):
        return RelevanceGrade(
            relevant_chunk_ids=[c["chunk_id"] for c in chunks],
            reasoning="all relevant",
        )

    return _grade


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
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
def mock_search_service():
    return MagicMock()


@pytest.fixture
def mock_document_service():
    return MagicMock()


@pytest.fixture
def client(mock_research_service, mock_search_service, mock_document_service):
    with (
        patch("app.main.ResearchService", return_value=mock_research_service),
        patch("app.main.SearchService", return_value=mock_search_service),
        patch("app.main.DocumentService", return_value=mock_document_service),
    ):
        app.dependency_overrides[get_research_service] = lambda: mock_research_service
        app.dependency_overrides[get_search_service] = lambda: mock_search_service
        app.dependency_overrides[get_document_service] = lambda: mock_document_service
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.clear()
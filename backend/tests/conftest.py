import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Settings have to be in place before app.main is imported: building the CORS
# middleware reads them, and that happens at import time, which is earlier than
# any fixture can run. Assigning rather than using setdefault also pins the
# values over whatever a developer keeps in backend/.env — otherwise the suite
# runs against their real provider and keys locally, and against nothing in CI,
# which is exactly how an import-time settings read reached CI unnoticed.
os.environ["FIRECRAWL_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["LLM_PROVIDER"] = "gemini"

from app.api.deps import get_document_service, get_research_service, get_search_service  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.domain.grading import AnswerQualityGrade, RelevanceGrade  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def keep_all_chunks_relevant():
    """side_effect for gemini.grade_relevance: the judge accepts every chunk.

    Pipeline tests exercise what happens after chunk selection, not the
    selection itself, so the judge has to be mocked explicitly. Otherwise
    MagicMock returns an empty relevant_chunk_ids, the node drops everything,
    and nothing ever reaches generation.
    """

    def _grade(question, chunks):
        return RelevanceGrade(
            relevant_chunk_ids=[c.chunk_id for c in chunks],
            reasoning="all relevant",
        )

    return _grade


@pytest.fixture
def answer_is_satisfactory():
    """return_value for llm.grade_answer_quality: the reviewer approves.

    A bare MagicMock used to pass this check by accident — its is_satisfactory
    attribute is truthy — while its reasoning was a MagicMock too. Anything that
    reads the reasoning then gets a mock where a string is declared, so pipeline
    tests state the verdict explicitly.
    """
    return AnswerQualityGrade(is_satisfactory=True, reasoning="on topic and useful")


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
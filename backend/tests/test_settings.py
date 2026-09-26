"""The suite must read its configuration from conftest, never from whatever
the developer happens to keep in backend/.env.

Without this, a test run means something different on every machine: locally it
picks up real keys and a real provider, in CI it picks up nothing at all. That
difference is what let an import-time settings read pass locally and fail CI.
"""

from app.core.config import get_settings


def test_settings_come_from_the_test_environment_not_the_env_file():
    settings = get_settings()

    assert settings.firecrawl_api_key == "test-key"
    assert settings.gemini_api_key == "test-key"
    # backend/.env may well select another provider; the suite must not see it.
    assert settings.llm_provider == "gemini"


def test_no_real_credential_reaches_the_suite():
    settings = get_settings()

    for value in (settings.firecrawl_api_key, settings.gemini_api_key):
        assert value == "test-key", "a real key leaked into the test settings"

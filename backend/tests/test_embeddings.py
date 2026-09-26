"""The embedding cache sits in front of a metered API and lives as long as the
process, so both what it stores and what it keys on have bitten before: it was
an unbounded dict, and it keyed on the text alone."""

from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.embeddings.gemini import (
    EMBED_CACHE_MAXSIZE,
    MAX_EMBED_RETRIES,
    GeminiEmbeddingsProvider,
)


@pytest.fixture
def embeddings():
    with patch(
        "app.infrastructure.embeddings.gemini.GoogleGenerativeAIEmbeddings"
    ) as cls:
        client = MagicMock()
        cls.return_value = client
        provider = GeminiEmbeddingsProvider()
        yield provider, client


def test_documents_are_embedded_once_and_then_served_from_cache(embeddings):
    provider, client = embeddings
    client.embed_documents.return_value = [[1.0], [2.0]]

    first = provider.embed_documents(["alpha", "beta"])
    second = provider.embed_documents(["alpha", "beta"])

    assert first == second == [[1.0], [2.0]]
    assert client.embed_documents.call_count == 1


def test_only_the_uncached_texts_are_sent_upstream(embeddings):
    provider, client = embeddings
    client.embed_documents.side_effect = [[[1.0]], [[2.0]]]

    provider.embed_documents(["alpha"])
    provider.embed_documents(["alpha", "beta"])

    assert client.embed_documents.call_args_list[1].args[0] == ["beta"]


def test_a_query_does_not_collide_with_the_same_text_as_a_document(embeddings):
    """Gemini embeds a passage and a question under different task types, so
    the same string has two different vectors. A cache keyed on the text alone
    served whichever was computed first."""
    provider, client = embeddings
    client.embed_documents.return_value = [[1.0, 0.0]]
    client.embed_query.return_value = [0.0, 1.0]

    as_document = provider.embed_documents(["gradient descent"])[0]
    as_query = provider.embed_query("gradient descent")

    assert as_document == [1.0, 0.0]
    assert as_query == [0.0, 1.0]
    client.embed_query.assert_called_once()


def test_queries_are_cached_too(embeddings):
    provider, client = embeddings
    client.embed_query.return_value = [0.5]

    assert provider.embed_query("q") == provider.embed_query("q") == [0.5]
    assert client.embed_query.call_count == 1


def test_rate_limited_call_is_retried(embeddings):
    provider, client = embeddings
    client.embed_query.side_effect = [RuntimeError("429 RESOURCE_EXHAUSTED"), [0.5]]

    with patch("app.infrastructure.embeddings.gemini.time.sleep"):
        assert provider.embed_query("q") == [0.5]

    assert client.embed_query.call_count == 2


def test_rate_limited_call_gives_up_after_the_retry_budget(embeddings):
    provider, client = embeddings
    client.embed_query.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

    with patch("app.infrastructure.embeddings.gemini.time.sleep"), pytest.raises(RuntimeError):
        provider.embed_query("q")

    assert client.embed_query.call_count == MAX_EMBED_RETRIES + 1


def test_a_non_rate_limit_error_is_not_retried(embeddings):
    provider, client = embeddings
    client.embed_query.side_effect = ValueError("bad input")

    with pytest.raises(ValueError):
        provider.embed_query("q")

    assert client.embed_query.call_count == 1


def test_the_cache_is_bounded(embeddings):
    """It outlives every request, and one gemini-embedding-001 vector is ~96 KiB,
    so an unbounded cache is a memory leak measured in hundreds of megabytes."""
    provider, client = embeddings
    client.embed_query.side_effect = lambda text: [float(len(text))]

    for i in range(EMBED_CACHE_MAXSIZE + 50):
        provider.embed_query(f"question {i}")

    assert len(provider._cache) <= EMBED_CACHE_MAXSIZE

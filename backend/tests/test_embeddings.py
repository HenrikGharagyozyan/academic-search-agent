"""The embedding cache sits in front of a metered API and lives as long as the
process, so both what it stores and what it keys on have bitten before: it was
an unbounded dict, and it keyed on the text alone."""

from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.embeddings.gemini import (
    EMBED_BATCH_SIZE,
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


# --- batching, so a rate-limited retry stays cheap -------------------------


def test_documents_are_sent_in_batches_of_the_providers_maximum(embeddings):
    provider, client = embeddings
    client.embed_documents.side_effect = lambda batch: [[float(len(t))] for t in batch]

    provider.embed_documents([f"text {i}" for i in range(250)])

    sizes = [len(call.args[0]) for call in client.embed_documents.call_args_list]
    assert sizes == [EMBED_BATCH_SIZE, EMBED_BATCH_SIZE, 50]


def test_a_rate_limited_batch_is_retried_alone(embeddings):
    """The whole call used to be wrapped in one retry, so a 429 on a later batch
    re-sent every text already embedded before it — twice, then failed."""
    provider, client = embeddings
    calls: list[int] = []

    def embed(batch):
        calls.append(len(batch))
        # fail once, on the second batch only
        if len(calls) == 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [[1.0] for _ in batch]

    client.embed_documents.side_effect = embed

    with patch("app.infrastructure.embeddings.gemini.time.sleep"):
        result = provider.embed_documents([f"text {i}" for i in range(150)])

    assert len(result) == 150
    # First batch sent once, second batch sent twice — not the first again.
    assert calls == [EMBED_BATCH_SIZE, 50, 50]


def test_work_done_before_a_failure_is_not_thrown_away(embeddings):
    """A hard failure part-way through still leaves the earlier batches cached,
    so a retry of the request does not pay for them again."""
    provider, client = embeddings
    calls = {"n": 0}

    def embed(batch):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("some permanent error")
        return [[1.0] for _ in batch]

    client.embed_documents.side_effect = embed
    texts = [f"text {i}" for i in range(150)]

    with pytest.raises(RuntimeError):
        provider.embed_documents(texts)

    # The first hundred are cached; asking for them alone makes no new call.
    calls["n"] = 0
    client.embed_documents.side_effect = lambda batch: [[1.0] for _ in batch]
    provider.embed_documents(texts[:100])
    assert calls["n"] == 0


def test_a_cached_batch_is_not_re_sent(embeddings):
    provider, client = embeddings
    client.embed_documents.side_effect = lambda batch: [[1.0] for _ in batch]
    texts = [f"text {i}" for i in range(150)]

    provider.embed_documents(texts)
    client.embed_documents.reset_mock()
    provider.embed_documents(texts)

    client.embed_documents.assert_not_called()

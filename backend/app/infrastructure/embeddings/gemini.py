import hashlib
import logging
import time

import cachetools
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
EMBED_CACHE_MAXSIZE = 2000
EMBED_CACHE_TTL_SECONDS = 3600

# Google's own per-request maximum. The batching matters less for the request
# count — the client splits large calls anyway — than for what a rate-limited
# retry costs: wrapping one call around a thousand texts meant a 429 on the
# seventh request re-sent the six hundred texts already embedded, twice, before
# giving up. Batching here keeps a retry to the batch that failed.
EMBED_BATCH_SIZE = 100

# Pause between batches. Zero by default because the retry handles a 429
# correctly now; raise it if a key's per-minute quota is the binding limit,
# which is cheaper than failing and falling back to unranked passages.
EMBED_BATCH_PAUSE_SECONDS = 0.0


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


class GeminiEmbeddingsProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self._cache: cachetools.TTLCache = cachetools.TTLCache(
            maxsize=EMBED_CACHE_MAXSIZE, ttl=EMBED_CACHE_TTL_SECONDS
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._cache_key("d", t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        # Each batch is cached as it arrives, so a failure part-way through
        # leaves the work already done behind it rather than discarding it.
        for start in range(0, len(missing_idx), EMBED_BATCH_SIZE):
            batch_idx = missing_idx[start : start + EMBED_BATCH_SIZE]
            fresh = self._call_with_retry(
                self._embeddings.embed_documents, [texts[i] for i in batch_idx]
            )
            for i, embedding in zip(batch_idx, fresh):
                self._cache[keys[i]] = embedding

            if EMBED_BATCH_PAUSE_SECONDS and start + EMBED_BATCH_SIZE < len(missing_idx):
                time.sleep(EMBED_BATCH_PAUSE_SECONDS)

        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._cache_key("q", text)
        if key not in self._cache:
            self._cache[key] = self._call_with_retry(self._embeddings.embed_query, text)
        return self._cache[key]

    @staticmethod
    def _cache_key(kind: str, text: str) -> str:
        return f"{kind}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _call_with_retry(fn, *args):
        last_exc: Exception | None = None
        for attempt in range(MAX_EMBED_RETRIES + 1):
            try:
                return fn(*args)
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt == MAX_EMBED_RETRIES:
                    raise
                wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "Gemini embeddings rate-limited, retrying in %.1fs (attempt %d/%d)",
                    wait, attempt + 1, MAX_EMBED_RETRIES,
                )
                time.sleep(wait)
        raise last_exc  # pragma: no cover
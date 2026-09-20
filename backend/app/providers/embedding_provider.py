import hashlib
import logging
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5.0


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


class EmbeddingProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._cache_key(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            fresh = self._call_with_retry(self._embeddings.embed_documents, missing_texts)
            for i, embedding in zip(missing_idx, fresh):
                self._cache[keys[i]] = embedding

        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if key not in self._cache:
            self._cache[key] = self._call_with_retry(self._embeddings.embed_query, text)
        return self._cache[key]

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
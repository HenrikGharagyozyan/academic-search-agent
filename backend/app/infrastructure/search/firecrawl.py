import logging
import threading
import time

import cachetools
from firecrawl import FirecrawlApp

from app.core.config import get_settings
from app.domain.search import ScrapedPage, SearchResult

logger = logging.getLogger(__name__)

SCRAPE_CACHE_TTL_SECONDS = 3600
SCRAPE_CACHE_MAXSIZE = 256

# Firecrawl counts searches and scrapes against one per-minute quota and
# refuses the rest. A burst of twelve concurrent scrapes hit "Consumed
# (req/min): 11, Remaining: 0" and the pages it refused were reported as
# unreadable — so a rate limit looked like a broken site, and which sources an
# answer was built from varied run to run for no visible reason. Retrying is
# the fix; the wait is the plan's, not ours to shorten.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_WAIT_SECONDS = 20.0


class RateLimited(Exception):
    """The provider refused a request for quota, not because of what was asked."""


class ScrapeRateLimited(RateLimited):
    """The provider refused a scrape for quota, not because the page is bad."""


class SearchRateLimited(RateLimited):
    """The provider refused a search for quota, not because nothing matched."""


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


def _date_filter(since_year: int) -> str:
    """Firecrawl passes `tbs` through to the search engine untouched, so this is
    Google's custom-date-range syntax: everything from 1 January of that year.

    A relative window (`qdr:y`, the past year) is the more common form but too
    tight for a literature question — a paper from eighteen months ago is still
    current research, and the interesting work in the reported case was two
    years old.
    """
    return f"cdr:1,cd_min:1/1/{since_year}"



class FirecrawlProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = FirecrawlApp(api_key=settings.firecrawl_api_key, timeout=30000)
        self._scrape_cache: cachetools.TTLCache = cachetools.TTLCache(
            maxsize=SCRAPE_CACHE_MAXSIZE, ttl=SCRAPE_CACHE_TTL_SECONDS
        )
        self._scrape_cache_lock = threading.Lock()

    def search(
        self, query: str, limit: int = 5, since_year: int | None = None
    ) -> list[SearchResult]:
        # Retried like a scrape: the searches share the scrapes' quota, and a
        # refused search silently costs the answer one of its planned angles —
        # or, with every query refused, the whole run.
        response = self._with_retry(
            lambda: self._client.search(
                query,
                limit=limit,
                tbs=_date_filter(since_year) if since_year else None,
            ),
            f"searching {query!r}",
            SearchRateLimited,
        )
        web_results = response.web or []

        return [
            SearchResult(
                title=item.title or "",
                url=item.url,
                snippet=item.description or "",
            )
            for item in web_results[:limit]
        ]

    def scrape(self, url: str) -> ScrapedPage:
        with self._scrape_cache_lock:
            cached = self._scrape_cache.get(url)
        if cached is not None:
            return cached

        response = self._scrape_with_retry(url)
        page = ScrapedPage(
            url=url,
            title=(response.metadata.title if response.metadata else "") or "",
            markdown=response.markdown or "",
        )

        if page.markdown:
            with self._scrape_cache_lock:
                self._scrape_cache[url] = page

        return page

    def _scrape_with_retry(self, url: str):
        return self._with_retry(
            lambda: self._client.scrape(url, formats=["markdown"]),
            f"scraping {url}",
            ScrapeRateLimited,
        )

    def _with_retry(self, call, action: str, exhausted: type[RateLimited]):
        """Runs ``call``, waiting out the provider's per-minute quota.

        Anything other than a rate limit is raised at once: a retry cannot fix
        a bad URL or a bad query, and would only spend more of the quota.
        """
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return call()
            except Exception as exc:
                if not _is_rate_limit(exc):
                    raise
                if attempt == RATE_LIMIT_RETRIES:
                    raise exhausted(
                        f"rate limited {action} after {attempt + 1} attempts"
                    ) from exc
                wait = RATE_LIMIT_WAIT_SECONDS * (attempt + 1)
                logger.warning(
                    "Firecrawl rate-limited %s, retrying in %.0fs (%d/%d)",
                    action, wait, attempt + 1, RATE_LIMIT_RETRIES,
                )
                time.sleep(wait)

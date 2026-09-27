import threading

import cachetools
from firecrawl import FirecrawlApp

from app.core.config import get_settings
from app.domain.search import ScrapedPage, SearchResult

SCRAPE_CACHE_TTL_SECONDS = 3600
SCRAPE_CACHE_MAXSIZE = 256


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
        response = self._client.search(
            query,
            limit=limit,
            tbs=_date_filter(since_year) if since_year else None,
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

        response = self._client.scrape(url, formats=["markdown"])
        page = ScrapedPage(
            url=url,
            title=(response.metadata.title if response.metadata else "") or "",
            markdown=response.markdown or "",
        )

        if page.markdown:
            with self._scrape_cache_lock:
                self._scrape_cache[url] = page

        return page
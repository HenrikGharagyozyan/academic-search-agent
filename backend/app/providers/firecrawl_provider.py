import threading

import cachetools
from firecrawl import FirecrawlApp
from pydantic import BaseModel

from app.core.config import get_settings

SCRAPE_CACHE_TTL_SECONDS = 3600
SCRAPE_CACHE_MAXSIZE = 256


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ScrapedPage(BaseModel):
    url: str
    title: str
    markdown: str


class FirecrawlProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = FirecrawlApp(api_key=settings.firecrawl_api_key, timeout=30000)
        self._scrape_cache: cachetools.TTLCache = cachetools.TTLCache(
            maxsize=SCRAPE_CACHE_MAXSIZE, ttl=SCRAPE_CACHE_TTL_SECONDS
        )
        self._scrape_cache_lock = threading.Lock()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        response = self._client.search(query, limit=limit)
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
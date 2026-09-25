"""Web search and page scraping."""

from typing import Protocol, runtime_checkable

from app.domain.search import ScrapedPage, SearchResult


@runtime_checkable
class SearchProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Returns at most ``limit`` result descriptors for the query."""

    def scrape(self, url: str) -> ScrapedPage:
        """Fetches a page and renders it to Markdown."""

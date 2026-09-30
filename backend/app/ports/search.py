"""Web search and page scraping."""

from typing import Protocol, runtime_checkable

from app.domain.search import ScrapedPage, SearchResult


@runtime_checkable
class SearchProvider(Protocol):
    def search(
        self, query: str, limit: int = 5, since_year: int | None = None
    ) -> list[SearchResult]:
        """Returns at most ``limit`` result descriptors for the query.

        ``since_year`` asks the engine to prefer pages published from that year
        onwards. It is a year rather than a provider-specific filter string so
        that translating it stays the adapter's problem.
        """

    def scrape(self, url: str) -> ScrapedPage:
        """Fetches a page and renders it to Markdown."""

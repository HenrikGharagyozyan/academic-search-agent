from app.providers.firecrawl_provider import FirecrawlProvider
from app.api.schemas.search import SearchResultItem


class SearchService:
    def __init__(self, provider: FirecrawlProvider | None = None) -> None:
        self._provider = provider or FirecrawlProvider()

    def search(self, query: str, limit: int) -> list[SearchResultItem]:
        raw_results = self._provider.search(query, limit=limit)

        return [
            SearchResultItem(title=r.title, url=r.url, snippet=r.snippet)
            for r in raw_results
        ]
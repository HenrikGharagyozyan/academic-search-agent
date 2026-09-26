from app.domain.search import SearchResult
from app.ports.search import SearchProvider


class SearchService:
    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    def search(self, query: str, limit: int) -> list[SearchResult]:
        return self._provider.search(query, limit=limit)

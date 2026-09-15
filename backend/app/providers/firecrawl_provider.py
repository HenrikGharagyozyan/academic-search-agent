from firecrawl import FirecrawlApp
from pydantic import BaseModel

from app.core.config import get_settings


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
        self._client = FirecrawlApp(api_key=settings.firecrawl_api_key)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        response = self._client.search(query, limit=limit)
        web_results = response.web or []

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in web_results[:limit]
        ]

    def scrape(self, url: str) -> ScrapedPage:
        response = self._client.scrape(url, formats=["markdown"])

        return ScrapedPage(
            url=url,
            title=(response.metadata.title if response.metadata else "") or "",
            markdown=response.markdown or "",
        )
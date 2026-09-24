from app.infrastructure.search.firecrawl import FirecrawlProvider
from app.domain.text.splitter import split_into_lines
from app.domain.documents import ParsedDocument


class DocumentService:
    def __init__(self, provider: FirecrawlProvider | None = None) -> None:
        self._provider = provider or FirecrawlProvider()

    def get_document(self, url: str) -> ParsedDocument:
        page = self._provider.scrape(url)
        lines = split_into_lines(page.markdown)

        return ParsedDocument(url=page.url, title=page.title, lines=lines)
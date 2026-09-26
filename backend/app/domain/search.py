"""What a web search and scrape yield, independent of which service performs them."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ScrapedPage(BaseModel):
    url: str
    title: str
    markdown: str

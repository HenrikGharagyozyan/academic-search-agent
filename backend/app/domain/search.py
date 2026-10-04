"""What a web search and scrape yield, independent of which service performs them."""

from pydantic import BaseModel

from app.domain.citation import SourceCitation


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ScrapedPage(BaseModel):
    url: str
    title: str
    markdown: str
    # Who wrote it and when, as the page declares; empty if it does not.
    citation: SourceCitation = SourceCitation()

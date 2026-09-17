from typing import TypedDict

from app.schemas.answer import Claim
from app.schemas.document import Chunk
from app.providers.firecrawl_provider import SearchResult


class ResearchState(TypedDict):
    question: str
    search_query: str
    search_results: list[SearchResult]
    chunks: list[Chunk]
    selected_chunks: list[Chunk]
    summary: str
    claims: list[Claim]
    conclusion: str
    retry_count: int
    evidence_sufficient: bool
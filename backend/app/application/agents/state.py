from typing import TypedDict

from app.domain.answers import Claim
from app.domain.documents import Chunk
from app.domain.search import SearchResult


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
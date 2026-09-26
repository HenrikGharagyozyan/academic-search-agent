import operator
from typing import Annotated, TypedDict

from app.domain.activity import ActivityStep
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
    # Appended to, never replaced: every node contributes, and a refine pass
    # adds to the record of the first one instead of erasing it.
    activity: Annotated[list[ActivityStep], operator.add]

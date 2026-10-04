import operator
from typing import Annotated, TypedDict

from app.domain.activity import ActivityStep
from app.domain.answers import Claim
from app.domain.documents import Chunk
from app.domain.search import SearchResult


class KeptAnswer(TypedDict):
    summary: str
    claims: list[Claim]
    conclusion: str
    # Every passage of that pass, so the kept claims can still be cited.
    chunks: list[Chunk]


class ResearchState(TypedDict):
    question: str
    search_query: str
    # The differently-aimed searches planned for this attempt, and the earliest
    # publication year to accept when the question asks for recent work.
    search_queries: list[str]
    since_year: int | None
    search_results: list[SearchResult]
    chunks: list[Chunk]
    selected_chunks: list[Chunk]
    summary: str
    claims: list[Claim]
    conclusion: str
    retry_count: int
    evidence_sufficient: bool
    # Why the last answer was judged insufficient, for the rewrite of the search
    # to aim at. Empty while nothing has fallen short.
    shortfall: str
    # The grader's problem code behind the shortfall ("" when there is none).
    shortfall_problem: str
    # The grounded answer of an earlier pass, set aside before a retry. A retry
    # can come back with nothing, and a thin answer is better than none.
    kept_answer: KeptAnswer | None
    # Appended to, never replaced: every node contributes, and a refine pass
    # adds to the record of the first one instead of erasing it.
    activity: Annotated[list[ActivityStep], operator.add]

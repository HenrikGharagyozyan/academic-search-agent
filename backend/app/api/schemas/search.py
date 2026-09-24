from pydantic import BaseModel, Field

from app.domain.search import SearchResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

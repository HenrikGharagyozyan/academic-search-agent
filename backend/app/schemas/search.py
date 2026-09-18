from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
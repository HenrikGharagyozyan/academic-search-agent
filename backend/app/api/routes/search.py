from fastapi import APIRouter

from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    service = SearchService()
    results = service.search(request.query, request.limit)

    return SearchResponse(query=request.query, results=results)
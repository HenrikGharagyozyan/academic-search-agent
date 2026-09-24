import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_search_service
from app.api.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        results = service.search(request.query, request.limit)
    except Exception:
        logger.warning("Search failed for query=%r", request.query, exc_info=True)
        raise HTTPException(
            status_code=502, detail="Search provider is unavailable"
        ) from None

    return SearchResponse(query=request.query, results=results)
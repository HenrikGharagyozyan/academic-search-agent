from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_research_service
from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.schemas.answer import Answer, AnswerRequest
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/v1", tags=["answer"])


@router.post("/answer", response_model=Answer)
def answer(
    request: AnswerRequest,
    service: ResearchService = Depends(get_research_service),
) -> Answer:
    try:
        return service.answer(request.question)
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ResearchServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
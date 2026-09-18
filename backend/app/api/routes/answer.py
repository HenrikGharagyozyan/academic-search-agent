from fastapi import APIRouter, HTTPException

from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.schemas.answer import Answer, AnswerRequest
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/v1", tags=["answer"])


@router.post("/answer", response_model=Answer)
def answer(request: AnswerRequest) -> Answer:
    service = ResearchService()
    try:
        return service.answer(request.question)
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ResearchServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
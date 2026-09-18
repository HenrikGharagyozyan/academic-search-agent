from fastapi import APIRouter

from app.schemas.answer import Answer, AnswerRequest
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/v1", tags=["answer"])


@router.post("/answer", response_model=Answer)
async def answer(request: AnswerRequest) -> Answer:
    service = ResearchService()
    return service.answer(request.question)
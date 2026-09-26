import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_research_service
from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.domain.answers import Answer
from app.api.schemas.answer import AnswerRequest
from app.application.services.research import ResearchService

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


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/answer/stream")
def answer_stream(
    request: AnswerRequest,
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    def event_generator():
        for event in service.stream_answer(request.question):
            yield _format_sse(event["event"], event["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
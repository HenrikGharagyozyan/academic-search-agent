import json
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_research_service
from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.domain.answers import Answer
from app.api.schemas.answer import AnswerRequest
from app.application.services.research import ResearchService

router = APIRouter(prefix="/api/v1", tags=["answer"])

# Well under the proxy's read timeout, so a silent stage never reaches it.
HEARTBEAT_SECONDS = 15
_DONE = object()


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
        # The pipeline runs on its own thread so that this one can speak while
        # it is silent. A reasoning model can think for minutes before the
        # answer arrives, and a proxy closes a stream that sends nothing for
        # its read timeout — nginx's two minutes cut off an answer mid-thought.
        events: queue.Queue = queue.Queue()

        def run() -> None:
            try:
                for event in service.stream_answer(request.question):
                    events.put(event)
            finally:
                events.put(_DONE)

        threading.Thread(target=run, daemon=True).start()

        while True:
            try:
                event = events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                # An SSE comment: it keeps the connection open, and every
                # client, ours included, skips it as carrying no data.
                yield ": ping\n\n"
                continue
            if event is _DONE:
                return
            yield _format_sse(event["event"], event["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
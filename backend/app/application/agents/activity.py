"""Recording what the agent does, for both the live feed and the stored trail."""

import logging
from urllib.parse import urlparse

from langgraph.config import get_stream_writer

from app.domain.activity import ActivityStep, StepKind

logger = logging.getLogger(__name__)


def short_host(url: str) -> str:
    """The bit of a URL worth putting in a progress line."""
    host = urlparse(url).netloc
    return host[4:] if host.startswith("www.") else host or url


def count(n: int, noun: str, plural: str | None = None) -> str:
    """"1 passage" / "4 passages" — these strings are read by a person."""
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def _stream_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        # No graph run around us: a unit test calling a node directly. The trail
        # is still collected, it just is not streamed anywhere.
        return None


class ActivityRecorder:
    """Collects steps for the graph state and forwards each one as it happens.

    A node's state update only arrives when the node returns, which is far too
    late for the slow ones: scraping six pages is a single node that can run for
    most of a minute, and "Reading sources" on screen for forty seconds tells
    the reader nothing. So each step is handed to LangGraph's custom stream writer
    the moment it is recorded, and the API turns those into SSE events.

    The writer resolves the run's config through a contextvar on every call, so
    it only works on the thread the node itself runs on. Calling it from a
    ThreadPoolExecutor worker raises "Called get_config outside of a runnable
    context", and capturing the writer in the node beforehand does not help
    because the closure reads the contextvar when invoked, not when created.
    A node that fans out therefore has to record from its own thread as results
    land — see retrieve_and_chunk_node.
    """

    def __init__(self, attempt: int = 0) -> None:
        self._steps: list[ActivityStep] = []
        self._attempt = attempt
        self._writer = _stream_writer()

    def record(
        self,
        kind: StepKind,
        label: str,
        *,
        url: str | None = None,
        title: str | None = None,
        detail: str | None = None,
    ) -> ActivityStep:
        step = ActivityStep(
            kind=kind,
            label=label,
            url=url,
            title=title,
            detail=detail,
            attempt=self._attempt,
        )
        self._steps.append(step)

        if self._writer is not None:
            # A broken feed must not fail the research run: the trail is
            # reporting, not the answer.
            try:
                self._writer(step.model_dump(mode="json"))
            except Exception:
                logger.warning("Could not stream activity step %s", kind, exc_info=True)

        return step

    @property
    def steps(self) -> list[ActivityStep]:
        return list(self._steps)

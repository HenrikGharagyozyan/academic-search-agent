"""The trail of what the agent did on the way to an answer."""

from typing import Literal

from pydantic import BaseModel

StepKind = Literal[
    "search",
    "source_found",
    "scrape_ok",
    "scrape_failed",
    "collect",
    "select",
    "grade_relevance",
    "generate",
    "verify",
    "grade_answer",
    "refine",
]


class ActivityStep(BaseModel):
    """One thing the agent did.

    The same shape is streamed while the run is in progress and stored on the
    finished Answer, so the interface renders one component for "what is
    happening now" and "what happened" rather than two that can disagree.

    `label` is written here rather than in the client because the pipeline is
    what knows the counts, and because STAGE_LABELS already establishes that
    the backend words its own progress.
    """

    kind: StepKind
    label: str
    url: str | None = None
    title: str | None = None
    detail: str | None = None
    # Which pass of the refine loop produced this step, so the trail can be
    # grouped when the agent searched more than once.
    attempt: int = 0

"""The answer the pipeline produces, and the claims it is made of."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.activity import ActivityStep

Confidence = Literal["high", "medium", "low"]
AnswerShape = Literal["direct", "survey"]


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: Confidence
    # The direction or approach this claim belongs to, when the evidence covers
    # several. Claims sharing a theme are meant to sit together, which is what
    # turns a flat list of facts into a survey of the area. Empty when the
    # evidence is about a single thing and grouping would be noise.
    theme: str = ""


class ClaimsResponse(BaseModel):
    """What the model is asked to return — before any verification."""

    # Declared first so it is written first: the model commits to what kind of
    # question this is before it writes a claim. As a paragraph of the prompt
    # the same decision was ignored — "what is energy" still came back as five
    # numbered sections — because nothing made the model take it.
    answer_shape: AnswerShape = Field(
        default="survey",
        description=(
            "Decide this FIRST, before writing anything else. "
            "'direct': the question asks what something is, what a formula means, "
            "or for one fact — there is one core answer. "
            "'survey': the question asks about research, challenges, approaches or "
            "a comparison — there are several independent aspects."
        ),
    )
    summary: str
    claims: list[Claim]
    conclusion: str


class AnswerEvidence(BaseModel):
    """A chunk as it is handed to the client, alongside the claims citing it."""

    chunk_id: str
    document_id: str
    text: str
    source_url: str
    title: str
    start_line: int
    end_line: int


class Answer(BaseModel):
    question: str
    summary: str
    claims: list[Claim]
    conclusion: str
    evidence: dict[str, AnswerEvidence]
    evidence_sufficient: bool
    # What the agent did to get here. Carried on the answer rather than only
    # streamed, so the trail survives a page reload and a non-streaming caller.
    activity: list[ActivityStep] = Field(default_factory=list)

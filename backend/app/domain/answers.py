"""The answer the pipeline produces, and the claims it is made of."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.activity import ActivityStep

Confidence = Literal["high", "medium", "low"]


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: Confidence


class ClaimsResponse(BaseModel):
    """What the model is asked to return — before any verification."""

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

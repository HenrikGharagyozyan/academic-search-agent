from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: Literal["high", "medium", "low"]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ClaimsResponse(BaseModel):
    summary: str
    claims: list[Claim]
    conclusion: str


class AnswerEvidence(BaseModel):
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
from pydantic import BaseModel


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: str  # "high" | "medium" | "low"


class ClaimsResponse(BaseModel):
    claims: list[Claim]


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
    claims: list[Claim]
    evidence: dict[str, AnswerEvidence]
from pydantic import BaseModel


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    confidence: str  # "high" | "medium" | "low"


class AnswerRequest(BaseModel):
    question: str


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
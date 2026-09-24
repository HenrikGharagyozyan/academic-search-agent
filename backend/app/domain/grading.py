from pydantic import BaseModel


class RelevanceGrade(BaseModel):
    relevant_chunk_ids: list[str]
    reasoning: str


class AnswerQualityGrade(BaseModel):
    is_satisfactory: bool
    reasoning: str
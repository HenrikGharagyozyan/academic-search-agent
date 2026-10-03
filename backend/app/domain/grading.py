from typing import Literal

from pydantic import BaseModel, Field


class RelevanceGrade(BaseModel):
    relevant_chunk_ids: list[str]
    reasoning: str


# Why an answer fell short, in the terms the next search can act on. A bare
# "not satisfactory" left the rewrite guessing, and it guessed a paraphrase.
Shortfall = Literal[
    "none",
    "off_topic",
    "too_thin",
    "missing_aspect",
    "non_substantive_evidence",
    "not_a_research_question",
]


class AnswerQualityGrade(BaseModel):
    is_satisfactory: bool
    reasoning: str
    problem: Shortfall = Field(
        default="none",
        description=(
            "'none' when satisfactory. Otherwise the main reason it is not: "
            "'off_topic' — the answer is about something other than what was "
            "asked; 'too_thin' — on topic but too little substance; "
            "'missing_aspect' — on topic but leaves out a part the question "
            "asks about; 'non_substantive_evidence' — built on navigation, ads "
            "or live-data widgets; 'not_a_research_question' — the question "
            "asks for something a literature search cannot answer."
        ),
    )
    missing: str = Field(
        default="",
        description=(
            "When not satisfactory: what the answer needed and did not have, "
            "named concretely enough to search for — the aspect, method, term "
            "or kind of source. Empty when satisfactory."
        ),
    )

"""Planning what to search for, before anything is searched.

A single query sent verbatim is the reason a well-studied topic comes back as
one paper: web search ranks by links, so the most-cited work and its mirrors
take every slot. Asking several deliberately different questions is what
surfaces the rest of the field.
"""

import re
from datetime import date

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """The searches to run for one question."""

    queries: list[str] = Field(default_factory=list)
    reasoning: str = ""


# Words that ask for the current state of a field rather than its foundations.
# Deliberately narrow: a false positive costs breadth by filtering out the
# canonical work, so only phrasings that clearly mean "now" are listed.
_RECENCY_TERMS = (
    "latest",
    "recent",
    "recently",
    "newest",
    "new",
    "current",
    "currently",
    "modern",
    "state of the art",
    "state-of-the-art",
    "emerging",
    "up to date",
    "up-to-date",
    "this year",
    "past year",
    "last year",
    "nowadays",
    "today",
    "trend",
    "trends",
    # Russian, since the interface accepts any language the model does.
    "последние",
    "последний",
    "недавние",
    "недавно",
    "новые",
    "новейшие",
    "současn",
    "актуальные",
    "сейчас",
    "тренд",
)

# A year this far back still counts as "recent" for a literature question.
RECENT_YEAR_WINDOW = 3

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def has_recency_intent(text: str, today: date | None = None) -> bool:
    """True when the question is asking about recent work rather than a topic.

    A year in the question counts only when it is recent: "research since 2024"
    is asking for new work, "the 1998 proof" is not.
    """
    if not text:
        return False

    lowered = text.lower()
    words = set(_WORD.findall(lowered))

    for term in _RECENCY_TERMS:
        if " " in term or "-" in term:
            if term in lowered:
                return True
        elif term in words or any(w.startswith(term) for w in words):
            return True

    current_year = (today or date.today()).year
    return any(
        current_year - RECENT_YEAR_WINDOW <= int(match.group()) <= current_year + 1
        for match in _YEAR.finditer(text)
    )

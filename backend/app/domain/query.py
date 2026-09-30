"""Planning what to search for, before anything is searched.

A single query sent verbatim is the reason a well-studied topic comes back as
one paper: web search ranks by links, so the most-cited work and its mirrors
take every slot. Asking several deliberately different questions is what
surfaces the rest of the field.

The opposite failure is just as bad and arrived as soon as expansion did. Asked
for different directions, a model generalises: "transmission matrix engineering"
became "inverse scattering theory", and the answer came back about the nonlinear
Schrödinger equation and the Born approximation — a neighbouring field that
shares vocabulary and answers a different question. Breadth is only useful
inside the subject that was asked about, so a planned query has to keep the
question's own terms, and one that does not is discarded rather than searched.
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

# Words that carry no subject: grammar, and the vocabulary of asking about
# research rather than of any particular research. "latest research on X" and "X"
# must anchor to the same thing, or the recency wording would dilute the anchor.
_GENERIC = frozenset(
    """
    a an and are as at be been by can do does for from had has have how in into is
    it its of on or than that the their there these this to use used using was were
    what when where which who why will with
    advances advance analysis applications application approach approaches
    development developments field findings introduction latest literature method
    methods modern new newest overview paper papers progress publication recent
    research results review reviews state studies study survey technique techniques
    theory topic trend trends understanding work works
    """.split()
)

# A planned query must share this fraction of the question's anchor terms. Half
# is what separated the useful expansions from the drifting ones on real output:
# "real-time transmission matrix characterization" keeps two of three, while
# "adaptive optics for imaging 2023" and "inverse scattering theory" keep none.
ANCHOR_OVERLAP = 0.5


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


def fold_plural(word: str) -> str:
    """"transformers" and "transformer" are one term. Nothing cleverer than
    that: a stemmer would also fold "mechanistic" onto "mechanism"."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def anchor_terms(question: str) -> frozenset[str]:
    """The words that say what the question is about, stripped of everything that
    only says it is a question about research."""
    return frozenset(
        fold_plural(word)
        for word in _WORD.findall(question.lower())
        if len(word) > 2 and word not in _GENERIC
    )


def stays_on_topic(
    query: str, anchors: frozenset[str], overlap: float = ANCHOR_OVERLAP
) -> bool:
    """True when a planned query is still about what was asked.

    Generalising is the drift this catches: a query that keeps none of the
    question's subject words is searching a different subject, however adjacent
    it looks to the model that wrote it.
    """
    if not anchors:
        return True

    required = max(1, round(len(anchors) * overlap))
    return len(anchor_terms(query) & anchors) >= required

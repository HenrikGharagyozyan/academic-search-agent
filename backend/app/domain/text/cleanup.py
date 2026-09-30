"""Strips bookkeeping that the model leaks into prose.

A claim carries its evidence ids in a dedicated field, and the interface draws
the citation markers from it. When the model also writes those ids into the
claim text, the reader gets sentences ending in a string of UUIDs. The prompt
forbids it; this removes whatever slips through anyway.

The model copies ids by hand, and it does not always copy them faithfully: a
real answer ended ".[evidence_id: 72cacfe-cb0a-...]", the first group one
character short of a UUID. A pattern that insisted on exactly 8-4-4-4-12 let
that through to the reader. So an id is recognised by its shape — hex groups
joined by hyphens — rather than its exact length, and anything in brackets that
is labelled as evidence goes whole, whatever it contains.
"""

import re

from app.domain.text.fences import outside_fences

# A UUID as the model writes it: five or more hex groups joined by hyphens, of
# roughly the right lengths. Loose enough for a dropped or doubled character, or
# two ids run together; five groups keep dates and version numbers out of it.
_UUID = r"(?<![\w-])[0-9a-fA-F]{4,12}(?:-[0-9a-fA-F]{2,12}){4,}(?![\w-])"
_LABEL = r"\bevidence[_ ]ids?\b"

# One id, or several joined by commas / "and", each optionally re-labelled.
_ID_LIST = rf"{_UUID}(?:\s*(?:,|and)\s*(?:{_LABEL}\s*:?\s*)?{_UUID})*"

# "(evidence_ids: a, b)" or "[evidence_id: a]" — the whole aside goes. Once the
# label and a colon open the brackets, whatever follows up to the closing one is
# bookkeeping, recognisable as an id or not.
_BRACKETED = re.compile(
    rf"\s*[(\[]\s*{_LABEL}\s*(?::[^()\[\]\n]*|{_ID_LIST}\s*)[)\]]", re.IGNORECASE
)

# "evidence_id: a" sitting bare in the sentence.
_BARE = re.compile(rf"\s*{_LABEL}\s*:?\s*{_ID_LIST}", re.IGNORECASE)

# A bare id with no label at all.
_LOOSE = re.compile(rf"\s*[(\[]?\s*{_UUID}\s*[)\]]?")

# Connectors and punctuation left dangling once an id is cut out.
_LEFTOVERS = [
    (re.compile(r"\(\s*\)|\[\s*\]"), ""),
    (re.compile(r"\s+(and|,)\s*,"), ","),
    (re.compile(r"\s+([,.;:])"), r"\1"),
    (re.compile(r"[ \t]{2,}"), " "),
]


# Anything that still looks like bookkeeping once the stripping is done.
_LEAK = re.compile(rf"{_LABEL}|{_UUID}", re.IGNORECASE)


def find_evidence_id_leak(text: str) -> str | None:
    """The first trace of an evidence id left in ``text``, or None if it is clean.

    Run on the finished text, fences included: stripping skips fenced blocks to
    keep diagrams intact, but an id inside a diagram is still an id the reader
    should never see.
    """
    match = _LEAK.search(text)
    return match.group() if match else None


def strip_evidence_ids(text: str) -> str:
    """Removes evidence ids the model wrote into prose, and tidies what is left.

    Fenced blocks are left alone: the tidying collapses runs of spaces, and in a
    diagram those runs are the drawing.
    """
    return outside_fences(text, _strip_from_prose)


def _strip_from_prose(text: str) -> str:
    # Each pattern eats the whitespace on both sides of the id, so cutting one
    # out of mid-sentence would run the surrounding words together
    # ("Adam <id> converges" -> "Adamconverges"). Substituting a space keeps the
    # word boundary; _LEFTOVERS then collapses whatever that leaves behind.
    cleaned = _BRACKETED.sub(" ", text)
    cleaned = _BARE.sub(" ", cleaned)
    cleaned = _LOOSE.sub(" ", cleaned)

    for pattern, replacement in _LEFTOVERS:
        cleaned = pattern.sub(replacement, cleaned)

    return cleaned.strip()

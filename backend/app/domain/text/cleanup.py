"""Strips bookkeeping that the model leaks into prose.

A claim carries its evidence ids in a dedicated field, and the interface draws
the citation markers from it. When the model also writes those ids into the
claim text, the reader gets sentences ending in a string of UUIDs. The prompt
forbids it; this removes whatever slips through anyway.
"""

import re

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_LABEL = r"evidence[_ ]ids?"

# One id, or several joined by commas / "and", each optionally re-labelled.
_ID_LIST = rf"{_UUID}(?:\s*(?:,|and)\s*(?:{_LABEL}\s*:?\s*)?{_UUID})*"

# "(evidence_ids: a, b)" or "[evidence_id: a]" — the whole aside goes.
_BRACKETED = re.compile(rf"\s*[(\[]\s*{_LABEL}\s*:?\s*{_ID_LIST}\s*[)\]]", re.IGNORECASE)

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


def strip_evidence_ids(text: str) -> str:
    """Removes evidence ids the model wrote into prose, and tidies what is left."""
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

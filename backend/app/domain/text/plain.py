"""Reducing a scraped string to something worth showing a reader.

A search result carries whatever the engine scraped as a title or a snippet, and
for many sites that is not prose: arXiv returns its metadata table, publishers
return their share-button rail, and both arrive as Markdown with links, images,
mailto: URLs and table pipes in them. Shown verbatim in the activity trail, that
is a wall of punctuation the reader cannot interpret.

Flattening the whole thing is not enough — it just yields a longer wall. What
makes this tractable is that the junk always follows the useful part: the title
comes first, then the metadata table, the author rail, the year filter. So the
string is cut at its first structural boundary and only the opening segment is
kept.
"""

import re

# Order matters: an image is a link with a bang in front, so it goes first, and
# the link rule keeps the visible text while dropping the target.
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^<>]{1,300}?>")
_BARE_URL = re.compile(r"(?:https?://|mailto:|www\.)\S+")
_EMPHASIS = re.compile(r"[*_`]{1,3}")

# Where the useful part ends: a line break, a table cell, or a new heading.
_BOUNDARY = re.compile(r"[\n|]+|#{1,6}")

_LEADING_BULLET = re.compile(r"^[\s\-*+•]+")
# arXiv prefixes its own field name onto the title.
_FIELD_PREFIX = re.compile(r"^(?:title|abstract|subjects)\s*:\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_LETTERS = re.compile(r"[^\W\d_]")

# Punctuation left dangling by the cut. A full stop is deliberately absent:
# it ends a sentence rather than marking a boundary we created.
_DEBRIS = " -–—:;,|"

# A label needs enough real words to mean anything; below this it is debris such
# as a lone "Share" or "Email" left over from a navigation rail.
MIN_LETTERS = 12


def to_label(text: str | None, limit: int = 110) -> str:
    """The first readable fragment of a scraped string, or "" if there is none.

    Returning "" rather than a shortened mess lets the caller omit the field
    instead of showing the reader something meaningless.
    """
    if not text:
        return ""

    flattened = _IMAGE.sub(" ", text)
    flattened = _LINK.sub(r"\1", flattened)
    flattened = _HTML_TAG.sub(" ", flattened)
    flattened = _BARE_URL.sub(" ", flattened)
    flattened = _EMPHASIS.sub("", flattened)

    for segment in _BOUNDARY.split(flattened):
        candidate = _clean_segment(segment)
        if len(_LETTERS.findall(candidate)) >= MIN_LETTERS:
            return _truncate(candidate, limit)

    return ""


def _clean_segment(segment: str) -> str:
    cleaned = _LEADING_BULLET.sub("", segment)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    cleaned = _FIELD_PREFIX.sub("", cleaned)
    return cleaned.strip(_DEBRIS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    cut = text.rfind(" ", 0, limit + 1)
    return text[: cut if cut > limit // 2 else limit].rstrip(_DEBRIS + ".") + "…"

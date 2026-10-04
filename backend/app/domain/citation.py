"""Who wrote a page and when, as the page itself declares it.

Publishers, preprint servers and proceedings put Google Scholar's citation
tags in every article's head: citation_author, citation_publication_date,
citation_date. Read from there, an attribution's year is the source's own
record. Left to the model, it was whatever the passage text or the model's
memory suggested, and the same question cited Cheng et al. as 2022 in one run
and 2023 in the next.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

# Most specific first: the date the work was published, then the date the page
# carries for it, then generic page dates. A page date is a weaker witness — a
# blog's modified date is not when its subject was published — so it is only
# used when nothing citation-specific is there.
_DATE_KEYS = (
    "citation_publication_date",
    "citation_date",
    "citation_online_date",
    "citation_cover_date",
    "dc.date",
    "dc_date",
    "dc.date.issued",
    "prism.publicationdate",
    "article:published_time",
    "published_time",
)
_AUTHOR_KEYS = ("citation_author", "dc.creator", "dc_creator")

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


class SourceCitation(BaseModel):
    """The authors and year a page declares; empty when it declares none."""

    authors: list[str] = []
    year: int | None = None


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [v for v in value if isinstance(v, str)]
    return []


def citation_from_metadata(metadata: Mapping[str, Any]) -> SourceCitation:
    """The page's own citation record, from its metadata tags."""
    lowered = {str(k).lower(): v for k, v in metadata.items()}

    authors: list[str] = []
    for key in _AUTHOR_KEYS:
        authors = [a.strip() for a in _values(lowered.get(key)) if a.strip()]
        if authors:
            break

    year = None
    for key in _DATE_KEYS:
        for value in _values(lowered.get(key)):
            if match := _YEAR.search(value):
                year = int(match.group(1))
                break
        if year is not None:
            break

    return SourceCitation(authors=authors, year=year)


def family_name(author: str) -> str:
    """"Resisi, Shachar" and "Shachar Resisi" are both Resisi."""
    author = author.strip()
    if "," in author:
        return author.split(",", 1)[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


def short_attribution(citation: SourceCitation) -> str:
    """"Resisi et al. (2019)", "Resisi and Bromberg (2019)", "Resisi (2019)";
    "" when the page declares no author."""
    names = [family_name(a) for a in citation.authors if family_name(a)]
    if not names:
        return ""
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{names[0]} et al."
    return f"{who} ({citation.year})" if citation.year else who

"""Holding an answer's attributions to the sources' own records.

The prompt shows each passage's authors and year and asks for them to be used,
but a model asked twice wrote "Cheng et al. (2022)" once and "(2023)" the
next time for the same source. Where a written attribution names an author of
a cited source, its year is that source's, set here rather than left to the
model.
"""

import re
from collections.abc import Sequence

from app.domain.citation import SourceCitation, family_name

# "Cheng et al. (2022)", "Cheng and Li (2022)", "Cheng (2022)", and the
# parenthetical forms "(Cheng et al., 2022)" and "(Cheng, 2022)".
_ATTRIBUTION = re.compile(
    r"(?P<who>\b(?P<surname>[A-Z][\w'’-]+)(?: et al\.?| and [A-Z][\w'’-]+)?)"
    r"(?P<sep>,? \(?)(?P<year>(?:1[89]|20)\d{2})\b"
)


def _years_for(surname: str, citations: Sequence[SourceCitation]) -> set[int]:
    """The years of the cited sources this surname is an author of."""
    wanted = surname.casefold()
    return {
        c.year
        for c in citations
        if c.year is not None
        and any(family_name(a).casefold() == wanted for a in c.authors)
    }


def correct_years(text: str, citations: Sequence[SourceCitation]) -> tuple[str, int]:
    """The text with each attribution's year set to its source's record, and
    how many were changed.

    A year is only replaced when it is unambiguous: the surname belongs to an
    author of exactly one cited year. An author of two cited works — a 2019
    preprint and its 2020 paper — keeps whichever of the two was written, and
    a name no cited source declares is left alone, since a passage may cite
    other work by name.
    """
    changed = 0

    def fix(match: re.Match) -> str:
        nonlocal changed
        years = _years_for(match["surname"], citations)
        written = int(match["year"])
        if written in years or len(years) != 1:
            return match.group(0)
        changed += 1
        return f"{match['who']}{match['sep']}{years.pop()}"

    return _ATTRIBUTION.sub(fix, text), changed

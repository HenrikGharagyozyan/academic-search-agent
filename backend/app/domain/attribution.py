"""Keeping a source's name out of a section heading."""

import re

# An attribution as it appears in a heading, with whatever joins it to the rest:
# "Adaptive correction — Jiang et al. (2026)", "Jiang et al. (2026): …".
# Only the unmistakable forms: several authors ("et al.", "and X") or a year in
# brackets. "QLoRA 2023 variants" names a method and a year, not a source.
_IN_THEME = re.compile(
    r"\s*[—–:|,-]?\s*\(?\b(?:"
    r"(?P<many>[A-Z][\w'’-]+(?: et al\.?| and [A-Z][\w'’-]+)),? \(?(?P<many_year>(?:1[89]|20)\d{2})\)?"
    r"|(?P<one>[A-Z][\w'’-]+) \((?P<one_year>(?:1[89]|20)\d{2})\)"
    r")\)?\s*[—–:|,-]?\s*"
)
_ENDS_SENTENCE = re.compile(r"[.!?]\s*$")


def move_attribution_out_of_theme(theme: str, text: str) -> tuple[str, str]:
    """The heading without attribution, and the claim carrying it.

    Theme and text are two fields the model fills separately, and an answer
    came back with "Jiang et al. (2026)" in a section heading and no name in
    the claim beneath it. A heading is a direction the section's claims share,
    so the attribution belongs to the claim: it is taken out of the heading
    and, if the claim does not already name that author, added to the end of
    the claim's first paragraph as "(Jiang et al., 2026)".
    """
    moved: list[tuple[str, str]] = []

    def take(match: re.Match) -> str:
        moved.append((match["many"] or match["one"], match["many_year"] or match["one_year"]))
        return " "

    heading = _IN_THEME.sub(take, theme).strip(" —–:|,-")
    if not moved:
        return theme, text

    first, sep, rest = text.partition("\n\n")
    for who, year in moved:
        surname = who.split()[0]
        if surname.casefold() in text.casefold():
            continue
        cite = f"({who}, {year})"
        if _ENDS_SENTENCE.search(first):
            stripped = first.rstrip()
            first = f"{stripped[:-1]} {cite}{stripped[-1]}"
        else:
            first = f"{first.rstrip()} {cite}"
    return heading, first + sep + rest

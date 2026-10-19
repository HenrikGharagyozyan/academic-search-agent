"""Splitting prose into sentences, for the chunker to cut between.

A full stop is not enough on its own: scientific text is full of "et al.",
"Fig. 3" and "Eq. (2)", and a formula can hold a full stop between its dollar
signs. A cut in any of those leaves a passage that starts mid-thought.
"""

import re

# Abbreviations that end in a full stop without ending the sentence, lowercase
# and without the stop.
_ABBREVIATIONS = frozenset(
    {
        "al", "approx", "cf", "ch", "dr", "e.g", "eq", "eqs", "etc", "fig",
        "figs", "i.e", "mr", "mrs", "ms", "no", "nos", "pp", "prof", "ref",
        "refs", "sec", "st", "vol", "vs",
    }
)

# End punctuation, any closing quotes or brackets after it, then whitespace,
# then something that can open a sentence.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[$]?[A-Z0-9$])")
_LAST_WORD = re.compile(r"(\S+)$")


def _ends_with_abbreviation(text: str) -> bool:
    match = _LAST_WORD.search(text)
    if not match:
        return False
    word = match.group(1).lower().rstrip(".").lstrip("(\"'“‘")
    # A single capital before the stop is an initial: "Cheng, A. Smith".
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def split_sentences(text: str) -> list[str]:
    """The sentences of ``text``, in order, each stripped of outer whitespace.

    Text with no sentence boundary — a heading, a table row, a formula —
    comes back as one sentence.
    """
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        stop = match.start() + 1
        head = text[start:stop]
        if _ends_with_abbreviation(head):
            continue
        # Inside $...$ a full stop is mathematics, not punctuation.
        if text.count("$", 0, stop) % 2:
            continue
        # The closing quotes and brackets belong to the sentence they close.
        end = match.end() - (len(match.group()) - len(match.group().rstrip()))
        sentences.append(text[start:end].strip())
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]

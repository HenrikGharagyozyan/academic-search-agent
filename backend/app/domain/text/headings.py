"""Turning a claim's theme into a section heading.

The interface shows each theme as a numbered heading — "1. Error rates and
sources" — and does the numbering itself, because only it knows which claims
survived verification. The model's part is the words, and it does not always
write them as a heading: the prompt's own examples were lowercase, so themes
arrived as "error rates and sources", and a model that numbers things on its
own would produce "1. 1. ...". This puts the theme into heading form whatever
shape it came in.
"""

import re

# Markdown heading marks, list numbering and bullets the model put in front:
# "### ", "1. ", "2) ", "- ".
_PREFIX = re.compile(r"^\s*(?:#+\s*|\d+\s*[.):]\s*|[-*•]\s+)+")
_TRAILING = re.compile(r"[\s.:;,]+$")


def tidy_theme(theme: str) -> str:
    """The theme as the words of a heading: no prefix, capital first letter.

    Only an all-lowercase first word is capitalised. "qLDPC codes" and
    "mRNA vaccines" are spelled that way on purpose, and the heading keeps them.
    """
    text = _TRAILING.sub("", _PREFIX.sub("", theme.strip()))
    if not text:
        return ""
    first = text.split(maxsplit=1)[0]
    if first.islower():
        text = text[0].upper() + text[1:]
    return text

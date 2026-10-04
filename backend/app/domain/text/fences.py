"""Keeping fenced blocks out of the text filters.

The filters that tidy model output are all about prose: restoring backslashes
inside maths, cutting evidence ids the model leaked into a sentence, collapsing
the whitespace that leaves behind. A fenced block is not prose. Its spacing is
the content — an ASCII diagram is drawn with it — and the collapse rule turned

    Physical qubits
          |
          v

into three lines indented by one space, which is not a diagram any more.

A fence is verbatim by definition, so the filters run around it rather than
through it.
"""

import re
from collections.abc import Callable

# Opening and closing ``` on their own, with whatever info string follows.
_FENCE = re.compile(r"^[ \t]*```.*$", re.MULTILINE)


def outside_fences(text: str, transform: Callable[[str], str]) -> str:
    """Applies ``transform`` to the parts of ``text`` that are not fenced.

    An unclosed fence protects everything after it, which is what a Markdown
    renderer does with it too, and is the safer way to be wrong.
    """
    if "```" not in text:
        return transform(text)

    out: list[str] = []
    cursor = 0
    inside = False

    for fence in _FENCE.finditer(text):
        segment = text[cursor : fence.start()]
        out.append(segment if inside else transform(segment))
        out.append(fence.group())
        cursor = fence.end()
        inside = not inside

    tail = text[cursor:]
    out.append(tail if inside else transform(tail))

    return "".join(out)

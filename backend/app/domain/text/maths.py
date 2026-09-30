"""Collapsing the three copies of every formula a Wikipedia page arrives with.

MediaWiki renders a formula as hidden MathML, a TeX annotation and a fallback
image. Converted to Markdown, all three survive, run together with no space:

    S=kBln⁡Ω{\\\\displaystyle S=k\\_{\\\\text{B}}\\\\ln \\\\Omega }![{\\displaystyle
    S=k_{\\text{B}}\\ln \\Omega }](https://wikimedia.org/api/rest_v1/media/math/…)

— the MathML's text with its structure gone ("kB" for k_B, "m2" for m²), the
annotation with Markdown's escaping on top of TeX's, then the image. A passage
about entropy is mostly this, and it is what the model is asked to read an
equation out of: the one clean copy is buried in an image's alt text, and the
first thing on the line is a flattened string that means something else.

The image's alt text is the TeX exactly as the page's author wrote it, so that
is the copy kept, as $...$ — which is also the notation the model is asked to
answer in.
"""

import re

# The fallback image. Its alt text is the formula; the host is what makes it
# safe to treat as one and not as an ordinary picture.
_MATH_IMAGE = re.compile(
    r"!\[\{\\(?P<style>displaystyle|textstyle)\s(?P<tex>.*?)\}\]"
    r"\(https://wikimedia\.org/api/rest_v1/media/math/render/[a-z]+/[0-9a-f]+\)",
    re.DOTALL,
)

# Where the escaped annotation in front of the image begins.
_ANNOTATION_START = re.compile(r"\{\\\\(?:displaystyle|textstyle)")

# The end of a Markdown link, which the flattened MathML is often glued to:
# `[extensive quantity](… "Extensive quantity")θ{\\textstyle …`.
_LINK_END = re.compile(r"""["'\w/]\)""")


def _duplicate_start(text: str, image_start: int) -> int:
    """Where the MathML text and the annotation in front of an image begin."""
    annotation = None
    for match in _ANNOTATION_START.finditer(text, 0, image_start):
        annotation = match
    if annotation is None or text[annotation.start() : image_start].count("\n") > 3:
        return image_start

    # The flattened MathML has no spaces in it, so it is the run of
    # non-whitespace immediately before the annotation — less anything that
    # belongs to a link the formula was set against.
    start = annotation.start()
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    run = text[start : annotation.start()]
    link_ends = list(_LINK_END.finditer(run))
    if link_ends:
        start += link_ends[-1].end()
    return start


def collapse_wiki_maths(markdown: str) -> str:
    """Replaces each MediaWiki formula triple with its TeX, as ``$...$``.

    A formula alone on its line is set as display maths, ``$$...$$``, the way
    the page set it. Text without such formulas is returned unchanged.
    """
    if "wikimedia.org/api/rest_v1/media/math" not in markdown:
        return markdown

    out: list[str] = []
    cursor = 0
    for image in _MATH_IMAGE.finditer(markdown):
        start = max(cursor, _duplicate_start(markdown, image.start()))
        out.append(markdown[cursor:start])

        tex = " ".join(image.group("tex").split())
        before = markdown[:start].rstrip(" \t")
        after = markdown[image.end() :].lstrip(" \t")
        alone = (not before or before.endswith("\n")) and (not after or after.startswith("\n"))
        delimiter = "$$" if alone and image.group("style") == "displaystyle" else "$"

        # Inline maths is glued to its neighbours in the source; a space keeps
        # "quantity$\theta$" from reading as one word.
        lead = "" if not out[-1] or out[-1][-1].isspace() or delimiter == "$$" else " "
        out.append(f"{lead}{delimiter}{tex}{delimiter}")
        cursor = image.end()

    out.append(markdown[cursor:])
    return "".join(out)

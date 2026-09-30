"""Telling whether two scraped pages are really the same document.

Six results from six domains looked like six sources and were one paper: the
arXiv version, the publisher's version, the lab's publication page, and two
aggregator stubs. Deduplicating by URL cannot see that, and comparing chunk
text exactly cannot either, because each site wraps the same words in its own
furniture.

Two signals, in this order.

The title is the document's identity, and academic mirrors reproduce it
verbatim: the same paper on arXiv and at the publisher carries the same title,
word for word. Where titles are available and specific enough, that settles it.

Content containment is the fallback, for pages whose titles were unusable.
Containment rather than Jaccard, because a mirror is often not the same size as
its original — an aggregator carries the abstract, the publisher the whole
article — and Jaccard punishes that difference exactly when the smaller document
is entirely inside the larger one.

Content alone is not enough, which a live run demonstrated: an arXiv abstract
page against the publisher's full article scored 0.41 and slipped through. The
arXiv page is mostly furniture — subject tags, submission history, bibliographic
tool links — so the abstract is a small part of it, and containment divides by
the smaller document including all of that. Their titles were identical.
"""

import re
from collections.abc import Sequence

# Word n-grams: long enough that shared phrasing is meaningful, short enough to
# survive a site reformatting the text around it.
SHINGLE_SIZE = 5

# How much of the smaller document has to appear in the larger one. Set from
# measurement, not taste. On the reported case: a publisher's article against an
# aggregator stub of its abstract scored 0.84, against the bare abstract 1.00 —
# while two genuinely different papers on the same topic scored 0.00, because a
# shared vocabulary produces almost no shared five-word runs. The threshold sits
# in the middle of that gap and is nowhere near either side.
MIRROR_THRESHOLD = 0.55

# How much of two titles must agree. Titles of distinct papers differ wildly, so this
# is deliberately strict — it is an identity check, not a similarity one.
TITLE_THRESHOLD = 0.85

# Below this a title is too generic to identify a document: "Adaptive Optics"
# would merge a textbook page with every paper that mentions it.
MIN_TITLE_WORDS = 4

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    """The set of word n-grams in the text, lowercased and stripped of markup."""
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()

    return frozenset(
        " ".join(words[i : i + size]) for i in range(len(words) - size + 1)
    )


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """How much of the smaller set appears in the larger one, from 0 to 1."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def title_words(title: str) -> frozenset[str]:
    """The identifying words of a title, or empty when it identifies nothing."""
    words = frozenset(_WORD.findall(title.lower()))
    return words if len(words) >= MIN_TITLE_WORDS else frozenset()


def same_title(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= TITLE_THRESHOLD


def find_mirrors(
    texts: Sequence[str],
    titles: Sequence[str] | None = None,
    threshold: float = MIRROR_THRESHOLD,
) -> dict[int, int]:
    """Maps each mirrored document to the index of the one kept in its place.

    Documents are compared longest first, so the version kept for a group is the
    fullest one — the journal article rather than the aggregator stub that
    reproduces its abstract. That is also the version worth reading: it has the
    method and the figures the stub omits.
    """
    fingerprints = [shingles(text) for text in texts]
    names = [title_words(t) for t in (titles or [""] * len(texts))]
    # Longest first, ties broken by original order so the result is stable.
    by_size = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))

    mirrors: dict[int, int] = {}
    kept: list[int] = []

    for index in by_size:
        canonical = next(
            (
                k
                for k in kept
                if same_title(names[index], names[k])
                or containment(fingerprints[index], fingerprints[k]) >= threshold
            ),
            None,
        )
        if canonical is None:
            kept.append(index)
        else:
            mirrors[index] = canonical

    return mirrors

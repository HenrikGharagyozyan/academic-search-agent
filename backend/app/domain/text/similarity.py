"""Telling whether two scraped pages are really the same document.

Six results from six domains looked like six sources and were one paper: the
arXiv version, the publisher's version, the lab's publication page, and two
aggregator stubs. Deduplicating by URL cannot see that, and comparing chunk
text exactly cannot either, because each site wraps the same words in its own
furniture.

Containment rather than Jaccard is the measure that works here. A mirror is
often not the same size as its original — an aggregator carries the abstract,
the publisher carries the whole article — and Jaccard punishes that difference
exactly when the smaller document is entirely inside the larger one. Overlap
against the smaller of the two is what "this is the same work" actually means.
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


def find_mirrors(
    texts: Sequence[str], threshold: float = MIRROR_THRESHOLD
) -> dict[int, int]:
    """Maps each mirrored document to the index of the one kept in its place.

    Documents are compared longest first, so the version kept for a group is the
    fullest one — the journal article rather than the aggregator stub that
    reproduces its abstract. That is also the version worth reading: it has the
    method and the figures the stub omits.
    """
    fingerprints = [shingles(text) for text in texts]
    # Longest first, ties broken by original order so the result is stable.
    by_size = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))

    mirrors: dict[int, int] = {}
    kept: list[int] = []

    for index in by_size:
        canonical = next(
            (
                k
                for k in kept
                if containment(fingerprints[index], fingerprints[k]) >= threshold
            ),
            None,
        )
        if canonical is None:
            kept.append(index)
        else:
            mirrors[index] = canonical

    return mirrors

"""Choosing which passages reach the model, once they are ranked.

Similarity alone answers "what is closest to the question", which is the wrong
question when the pages differ wildly in size. A journal article yields a
hundred passages and an aggregator stub yields one, so ranking by similarity
puts the long page's passages in nearly every slot — not because it is more
relevant, but because it has more entries in the draw.
"""

from collections.abc import Sequence

from app.domain.documents import Chunk


def spread_across_sources(chunks: Sequence[Chunk], limit: int) -> list[Chunk]:
    """Takes ``limit`` passages, giving every source a turn before any repeats.

    Sources are visited in the order of their best-ranked passage, so the most
    relevant page still leads, and within a source the ranking is preserved. A
    source that runs out is skipped, so a long page still fills the remaining
    slots once the others are exhausted — breadth first, then depth.

    A quota was tried first and could not work. Capping each source's share
    relies on the ranked list containing the other sources at all, and when one
    page holds a hundred of a hundred and thirty passages it fills the list
    before the cap is ever consulted. Measured on a live run: 25 slots, quota of
    10, and all 25 went to one page.
    """
    if limit <= 0:
        return []

    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_url, []).append(chunk)

    taken: list[Chunk] = []
    # dict preserves insertion order, which is the order the sources first
    # appear in the ranking — best source first.
    while len(taken) < limit and any(by_source.values()):
        for queue in by_source.values():
            if not queue:
                continue
            taken.append(queue.pop(0))
            if len(taken) >= limit:
                break

    return taken


def source_count(chunks: Sequence[Chunk]) -> int:
    """How many distinct pages the passages came from."""
    return len({chunk.source_url for chunk in chunks})

"""Choosing which passages reach the model, once they are ranked.

Similarity alone answers "what is closest to the question", which is the wrong
question when several sources say the same thing. One paper mirrored across
four domains scores four times, and its passages can take the whole context
window while the rest of the field waits outside.
"""

from collections.abc import Sequence

from app.domain.documents import Chunk


def cap_per_source(
    chunks: Sequence[Chunk], limit: int, max_per_source: int
) -> list[Chunk]:
    """Takes the best ``limit`` chunks, with no source exceeding its quota.

    Chunks arrive ranked, best first, and that order is preserved. A source at
    its quota is skipped rather than truncating the list, and if the quota rules
    leave room unused the skipped chunks fill it in rank order — running short
    would trade one bias for a worse one, less evidence overall.
    """
    if limit <= 0 or max_per_source <= 0:
        return []

    taken: list[Chunk] = []
    overflow: list[Chunk] = []
    per_source: dict[str, int] = {}

    for chunk in chunks:
        if len(taken) >= limit:
            break
        if per_source.get(chunk.source_url, 0) >= max_per_source:
            overflow.append(chunk)
            continue
        per_source[chunk.source_url] = per_source.get(chunk.source_url, 0) + 1
        taken.append(chunk)

    if len(taken) < limit:
        taken.extend(overflow[: limit - len(taken)])

    return taken


def source_count(chunks: Sequence[Chunk]) -> int:
    """How many distinct pages the passages came from."""
    return len({chunk.source_url for chunk in chunks})

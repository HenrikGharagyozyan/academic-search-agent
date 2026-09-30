"""Ranking passages by the question's words, for when embeddings are unavailable.

The fallback used to be "the first N passages", and with passages then dealt out
one source at a time that meant the first four chunks of every page: titles,
navigation, infoboxes, the opening paragraph. Nothing in that choice looks at
the question. On "what is entropy" the passage stating S = k_B ln Ω — present
on the page, a few hundred lines down — reached neither the judge nor the
model, and the answer was written from introductions.

This is not an edge case. The free embedding quota is a thousand texts a day
and one question has six to nine hundred passages, so after the first question
of the day every run lands here.

BM25 over the question's subject words is a long way short of an embedding, and
a long way better than position: a passage about the thing asked, wherever it
sits in the page, outranks a cookie notice at the top of it.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.domain.query import anchor_terms

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# The usual BM25 constants: how quickly a repeated term stops adding, and how
# strongly a long passage is discounted for having more words to match with.
_K1 = 1.5
_B = 0.75


def _stem(word: str) -> str:
    """Folds a plural onto its singular, which is all the matching needs."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _terms(text: str) -> list[str]:
    return [_stem(word) for word in _WORD.findall(text.lower())]


def rank_by_terms(question: str, texts: Sequence[str]) -> list[int]:
    """Indices of ``texts``, most relevant to the question first.

    Every index is returned; passages sharing no subject word with the question
    keep their original order at the end, so the result is a full ranking and a
    caller can cut it wherever it likes.
    """
    query = {_stem(term) for term in anchor_terms(question)}
    if not query or not texts:
        return list(range(len(texts)))

    documents = [Counter(_terms(text)) for text in texts]
    lengths = [sum(doc.values()) for doc in documents]
    average = (sum(lengths) / len(lengths)) or 1.0

    containing = {term: sum(1 for doc in documents if term in doc) for term in query}
    total = len(documents)

    def score(doc: Counter, length: int) -> float:
        result = 0.0
        for term in query:
            frequency = doc.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (total - containing[term] + 0.5) / (containing[term] + 0.5))
            norm = frequency + _K1 * (1 - _B + _B * length / average)
            result += idf * frequency * (_K1 + 1) / norm
        return result

    scores = [score(doc, length) for doc, length in zip(documents, lengths)]
    # Stable: equal scores, the unmatched passages among them, stay in page order.
    return sorted(range(total), key=lambda i: -scores[i])

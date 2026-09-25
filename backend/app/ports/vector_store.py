"""Similarity search over the chunks of a single request."""

from typing import Protocol, runtime_checkable

from app.domain.documents import Chunk


@runtime_checkable
class VectorStore(Protocol):
    def select_relevant_chunks(
        self, question: str, chunks: list[Chunk], top_k: int = 15
    ) -> list[Chunk]:
        """Returns the ``top_k`` chunks closest to the question, best first."""

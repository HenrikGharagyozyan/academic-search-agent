"""Text embeddings for semantic chunk selection."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingsProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds passages for storage. Vendors may use a different task type
        here than for queries, so the two are separate calls by design."""

    def embed_query(self, text: str) -> list[float]:
        """Embeds a question for retrieval."""

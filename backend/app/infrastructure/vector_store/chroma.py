import logging
import uuid

import chromadb

from app.infrastructure.embeddings.gemini import GeminiEmbeddingsProvider
from app.domain.documents import Chunk
from app.domain.text.lexical import rank_by_terms
from app.ports.embeddings import EmbeddingsProvider

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, embedding_provider: EmbeddingsProvider | None = None) -> None:
        self._embeddings = embedding_provider or GeminiEmbeddingsProvider()
        self._client = chromadb.EphemeralClient()

    def select_relevant_chunks(
        self, question: str, chunks: list[Chunk], top_k: int = 15
    ) -> list[Chunk]:
        if not chunks:
            return []

        try:
            return self._select_by_similarity(question, chunks, top_k)
        except Exception:
            # Not rare: the free embedding quota covers about one question a
            # day. Ranking by the question's words keeps the selection about
            # the question; taking the first passages did not look at it at all.
            logger.warning(
                "Embedding/vector search failed, ranking %d chunks by the "
                "question's terms instead",
                len(chunks),
                exc_info=True,
            )
            order = rank_by_terms(question, [c.text for c in chunks])
            return [chunks[i] for i in order[:top_k]]

    def _select_by_similarity(
        self, question: str, chunks: list[Chunk], top_k: int
    ) -> list[Chunk]:
        collection_name = f"chunks_{uuid.uuid4().hex}"
        collection = self._client.create_collection(name=collection_name)

        try:
            texts = [c.text for c in chunks]
            embeddings = self._embeddings.embed_documents(texts)
            ids = [c.chunk_id for c in chunks]

            collection.add(ids=ids, embeddings=embeddings, documents=texts)

            query_embedding = self._embeddings.embed_query(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(chunks)),
            )

            matched_ids = results["ids"][0]
            chunks_by_id = {c.chunk_id: c for c in chunks}

            return [chunks_by_id[cid] for cid in matched_ids if cid in chunks_by_id]
        finally:
            self._client.delete_collection(name=collection_name)
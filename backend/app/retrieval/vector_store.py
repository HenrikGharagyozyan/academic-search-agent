import uuid

import chromadb

from app.providers.embedding_provider import EmbeddingProvider
from app.schemas.document import Chunk


class ChunkVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._embeddings = embedding_provider or EmbeddingProvider()
        self._client = chromadb.EphemeralClient()

    def select_relevant_chunks(
        self, question: str, chunks: list[Chunk], top_k: int = 15
    ) -> list[Chunk]:
        if not chunks:
            return []

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
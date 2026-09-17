from unittest.mock import MagicMock

from app.retrieval.vector_store import ChunkVectorStore
from app.schemas.document import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        start_line=1,
        end_line=1,
        source_url="https://example.com",
        title="Example",
    )


def test_select_relevant_chunks_ranks_by_similarity():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0],  # chunk1 — far form query
        [0.0, 1.0],  # chunk2 — matches the request
    ]
    mock_embeddings.embed_query.return_value = [0.0, 1.0]

    store = ChunkVectorStore(embedding_provider=mock_embeddings)
    chunks = [make_chunk("c1", "irrelevant text"), make_chunk("c2", "relevant text")]

    result = store.select_relevant_chunks("some query", chunks, top_k=1)

    assert len(result) == 1
    assert result[0].chunk_id == "c2"


def test_select_relevant_chunks_empty_input():
    mock_embeddings = MagicMock()
    store = ChunkVectorStore(embedding_provider=mock_embeddings)

    assert store.select_relevant_chunks("query", [], top_k=5) == []
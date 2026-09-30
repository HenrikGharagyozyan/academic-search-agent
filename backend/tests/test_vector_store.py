from unittest.mock import MagicMock

from app.infrastructure.vector_store.chroma import ChromaVectorStore
from app.domain.documents import Chunk


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

    store = ChromaVectorStore(embedding_provider=mock_embeddings)
    chunks = [make_chunk("c1", "irrelevant text"), make_chunk("c2", "relevant text")]

    result = store.select_relevant_chunks("some query", chunks, top_k=1)

    assert len(result) == 1
    assert result[0].chunk_id == "c2"


def test_select_relevant_chunks_empty_input():
    mock_embeddings = MagicMock()
    store = ChromaVectorStore(embedding_provider=mock_embeddings)

    assert store.select_relevant_chunks("query", [], top_k=5) == []

def test_select_relevant_chunks_falls_back_on_embedding_failure():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = RuntimeError("quota exceeded")

    store = ChromaVectorStore(embedding_provider=mock_embeddings)
    chunks = [make_chunk("c1", "text one"), make_chunk("c2", "text two")]

    result = store.select_relevant_chunks("query", chunks, top_k=1)

    # Nothing here matches the question, so page order is all there is.
    assert [c.chunk_id for c in result] == ["c1"]


def test_the_fallback_ranks_by_the_question_instead_of_by_position():
    """The first passages of a page are its title and navigation. When the
    embedding quota is spent the selection must still be about the question."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

    store = ChromaVectorStore(embedding_provider=mock_embeddings)
    chunks = [
        make_chunk("nav", "Main page Contents Current events Random article Donate"),
        make_chunk("intro", "This article is about a concept in physics."),
        make_chunk("formula", "Boltzmann's entropy is S = k_B ln Ω, counting microstates."),
    ]

    result = store.select_relevant_chunks("What is entropy?", chunks, top_k=1)

    assert [c.chunk_id for c in result] == ["formula"]

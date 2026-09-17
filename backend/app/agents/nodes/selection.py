from app.agents.state import ResearchState
from app.retrieval.vector_store import ChunkVectorStore

TOP_K_CHUNKS = 15


def select_relevant_chunks_node(
    state: ResearchState, vector_store: ChunkVectorStore
) -> dict:
    selected = vector_store.select_relevant_chunks(
        state["question"], state["chunks"], top_k=TOP_K_CHUNKS
    )
    return {"selected_chunks": selected}
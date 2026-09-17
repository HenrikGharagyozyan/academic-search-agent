from app.agents.state import ResearchState
from app.retrieval.vector_store import ChunkVectorStore
from app.agents.constants import TOP_K_CHUNKS


def select_relevant_chunks_node(
    state: ResearchState, vector_store: ChunkVectorStore
) -> dict:
    selected = vector_store.select_relevant_chunks(
        state["question"], state["chunks"], top_k=TOP_K_CHUNKS
    )
    return {"selected_chunks": selected}
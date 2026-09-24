from app.application.agents.constants import TOP_K_CHUNKS
from app.application.agents.state import ResearchState
from app.ports.vector_store import VectorStore


def select_relevant_chunks_node(state: ResearchState, vector_store: VectorStore) -> dict:
    selected = vector_store.select_relevant_chunks(
        state["question"], state["chunks"], top_k=TOP_K_CHUNKS
    )
    return {"selected_chunks": selected}

from app.application.agents.state import ResearchState
from app.infrastructure.vector_store.chroma import ChromaVectorStore
from app.application.agents.constants import TOP_K_CHUNKS


def select_relevant_chunks_node(
    state: ResearchState, vector_store: ChromaVectorStore
) -> dict:
    selected = vector_store.select_relevant_chunks(
        state["question"], state["chunks"], top_k=TOP_K_CHUNKS
    )
    return {"selected_chunks": selected}
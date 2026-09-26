from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import TOP_K_CHUNKS
from app.application.agents.state import ResearchState
from app.ports.vector_store import VectorStore


def select_relevant_chunks_node(state: ResearchState, vector_store: VectorStore) -> dict:
    chunks = state["chunks"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    selected = vector_store.select_relevant_chunks(
        state["question"], chunks, top_k=TOP_K_CHUNKS
    )

    recorder.record(
        "select",
        f"Ranked {count(len(chunks), 'passage')}, kept the closest {len(selected)}",
        detail="by embedding similarity to the question",
    )

    return {"selected_chunks": selected, "activity": recorder.steps}

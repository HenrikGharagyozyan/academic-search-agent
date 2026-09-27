from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import TOP_K_CHUNKS
from app.application.agents.state import ResearchState
from app.domain.selection import source_count, spread_across_sources
from app.ports.vector_store import VectorStore


def select_relevant_chunks_node(state: ResearchState, vector_store: VectorStore) -> dict:
    chunks = state["chunks"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    # The full ranking, not the top TOP_K: the passages are already embedded, so
    # ranking all of them costs nothing, and a truncated list is how one large
    # page ends up being the only source represented at all.
    ranked = vector_store.select_relevant_chunks(
        state["question"], chunks, top_k=len(chunks)
    )
    selected = spread_across_sources(ranked, limit=TOP_K_CHUNKS)

    sources = source_count(selected)
    recorder.record(
        "select",
        f"Ranked {count(len(chunks), 'passage')}, kept {len(selected)} "
        f"from {count(sources, 'source')}",
        detail="each source gets a turn before any page repeats",
    )

    return {"selected_chunks": selected, "activity": recorder.steps}

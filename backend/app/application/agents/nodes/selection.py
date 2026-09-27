import math

from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import (
    CHUNK_OVERSAMPLE,
    MAX_SOURCE_SHARE,
    TOP_K_CHUNKS,
)
from app.application.agents.state import ResearchState
from app.domain.selection import cap_per_source, source_count
from app.ports.vector_store import VectorStore


def select_relevant_chunks_node(state: ResearchState, vector_store: VectorStore) -> dict:
    chunks = state["chunks"]
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    # Oversampled on purpose: the quota below discards some of what similarity
    # ranked highest, and asking for exactly TOP_K would leave the context short
    # by however many it discarded.
    ranked = vector_store.select_relevant_chunks(
        state["question"], chunks, top_k=TOP_K_CHUNKS * CHUNK_OVERSAMPLE
    )

    max_per_source = max(1, math.ceil(TOP_K_CHUNKS * MAX_SOURCE_SHARE))
    selected = cap_per_source(ranked, limit=TOP_K_CHUNKS, max_per_source=max_per_source)

    sources = source_count(selected)
    recorder.record(
        "select",
        f"Ranked {count(len(chunks), 'passage')}, kept {len(selected)} "
        f"from {count(sources, 'source')}",
        detail=f"no source may take more than {max_per_source} of {TOP_K_CHUNKS} slots",
    )

    return {"selected_chunks": selected, "activity": recorder.steps}

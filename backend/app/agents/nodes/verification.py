from app.agents.constants import MAX_RETRIES, ROUTE_END, ROUTE_REFINE
from app.agents.state import ResearchState


def verify_evidence_node(state: ResearchState) -> dict:
    chunks_by_id = {c.chunk_id: c for c in state["chunks"]}

    grounded_claims = [
        claim
        for claim in state["claims"]
        if any(eid in chunks_by_id for eid in claim.evidence_ids)
    ]

    sufficient = len(grounded_claims) > 0
    return {"evidence_sufficient": sufficient}


def should_refine(state: ResearchState) -> str:
    if state["evidence_sufficient"]:
        return ROUTE_END
    if state["retry_count"] >= MAX_RETRIES:
        return ROUTE_END
    return ROUTE_REFINE
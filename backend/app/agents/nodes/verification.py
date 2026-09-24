from app.agents.constants import MAX_RETRIES, ROUTE_END, ROUTE_REFINE
from app.agents.state import ResearchState
from app.domain.answers import Claim


def verify_evidence_node(state: ResearchState) -> dict:
    valid_ids = {c.chunk_id for c in state["selected_chunks"]}

    grounded_claims: list[Claim] = []
    for claim in state["claims"]:
        valid_evidence_ids = [eid for eid in claim.evidence_ids if eid in valid_ids]
        if valid_evidence_ids:
            grounded_claims.append(
                claim.model_copy(update={"evidence_ids": valid_evidence_ids})
            )

    return {
        "claims": grounded_claims,
        "evidence_sufficient": len(grounded_claims) > 0,
    }


def should_refine(state: ResearchState) -> str:
    if state["evidence_sufficient"]:
        return ROUTE_END
    if state["retry_count"] >= MAX_RETRIES:
        return ROUTE_END
    return ROUTE_REFINE
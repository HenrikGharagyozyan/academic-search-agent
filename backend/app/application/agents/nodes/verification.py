from app.application.agents.activity import ActivityRecorder, count
from app.application.agents.constants import MAX_RETRIES, ROUTE_END, ROUTE_REFINE
from app.application.agents.state import ResearchState
from app.domain.answers import Claim


def verify_evidence_node(state: ResearchState) -> dict:
    valid_ids = {c.chunk_id for c in state["selected_chunks"]}
    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))

    grounded_claims: list[Claim] = []
    trimmed = 0
    for claim in state["claims"]:
        valid_evidence_ids = [eid for eid in claim.evidence_ids if eid in valid_ids]
        if valid_evidence_ids:
            trimmed += len(claim.evidence_ids) - len(valid_evidence_ids)
            grounded_claims.append(
                claim.model_copy(update={"evidence_ids": valid_evidence_ids})
            )

    dropped = len(state["claims"]) - len(grounded_claims)
    details = []
    if dropped:
        details.append(f"{count(dropped, 'claim')} dropped as ungrounded")
    if trimmed:
        details.append(f"{count(trimmed, 'invented citation')} removed")

    recorder.record(
        "verify",
        f"Checked {count(len(state['claims']), 'claim')} against the passages, "
        f"{len(grounded_claims)} hold up",
        detail=", ".join(details) or None,
    )

    return {
        "claims": grounded_claims,
        "evidence_sufficient": len(grounded_claims) > 0,
        "activity": recorder.steps,
    }


def should_refine(state: ResearchState) -> str:
    if state["evidence_sufficient"]:
        return ROUTE_END
    if state["retry_count"] >= MAX_RETRIES:
        return ROUTE_END
    return ROUTE_REFINE

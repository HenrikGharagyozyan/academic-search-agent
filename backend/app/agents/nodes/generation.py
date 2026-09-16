from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider


def generate_claims_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["chunks"]:
        return {"claims": []}

    evidence_chunks = [
        {"chunk_id": c.chunk_id, "text": c.text} for c in state["chunks"]
    ]
    claims = gemini.generate_claims(state["question"], evidence_chunks)

    return {"claims": claims}
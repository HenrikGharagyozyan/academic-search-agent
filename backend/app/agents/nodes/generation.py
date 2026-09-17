import logging

from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


def generate_claims_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["selected_chunks"]:
        return {"claims": []}

    evidence_chunks = [
        {"chunk_id": c.chunk_id, "text": c.text} for c in state["selected_chunks"]
    ]

    try:
        claims = gemini.generate_claims(state["question"], evidence_chunks)
    except Exception:
        logger.warning("Failed to generate claims, returning empty", exc_info=True)
        claims = []

    return {"claims": claims}
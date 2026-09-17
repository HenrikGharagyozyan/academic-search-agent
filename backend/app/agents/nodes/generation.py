import logging

from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


def generate_claims_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["selected_chunks"]:
        return {"summary": "", "claims": [], "conclusion": ""}

    evidence_chunks = [
        {"chunk_id": c.chunk_id, "text": c.text} for c in state["selected_chunks"]
    ]

    try:
        result = gemini.generate_answer(state["question"], evidence_chunks)
    except Exception:
        logger.warning("Failed to generate answer, returning empty", exc_info=True)
        return {"summary": "", "claims": [], "conclusion": ""}

    return {
        "summary": result.summary,
        "claims": result.claims,
        "conclusion": result.conclusion,
    }
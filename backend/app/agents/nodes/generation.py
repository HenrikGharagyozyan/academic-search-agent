import logging

from app.agents.state import ResearchState
from app.providers.gemini_provider import GeminiProvider
from app.providers.latex import restore_latex
from app.providers.text_cleanup import strip_evidence_ids

logger = logging.getLogger(__name__)


def _presentable(text: str) -> str:
    """Turns raw model text into what the reader should actually see."""
    return strip_evidence_ids(restore_latex(text))


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
        "summary": _presentable(result.summary),
        "claims": [
            claim.model_copy(update={"text": _presentable(claim.text)})
            for claim in result.claims
        ],
        "conclusion": _presentable(result.conclusion),
    }
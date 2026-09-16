import logging

from app.agents.graph import build_research_graph
from app.providers.firecrawl_provider import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider
from app.schemas.answer import Answer, AnswerEvidence
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        firecrawl: FirecrawlProvider | None = None,
        gemini: GeminiProvider | None = None,
    ) -> None:
        self._graph = build_research_graph(firecrawl, gemini)

    def answer(self, question: str) -> Answer:
        result = self._graph.invoke(
            {"question": question, "search_results": [], "chunks": [], "claims": []}
        )

        chunks: list[Chunk] = result["chunks"]
        claims = result["claims"]

        chunks_by_id = {c.chunk_id: c for c in chunks}
        used_ids = {eid for claim in claims for eid in claim.evidence_ids}

        evidence = {
            eid: AnswerEvidence(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                source_url=c.source_url,
                title=c.title,
                start_line=c.start_line,
                end_line=c.end_line,
            )
            for eid in used_ids
            if (c := chunks_by_id.get(eid)) is not None
        }

        return Answer(question=question, claims=claims, evidence=evidence)
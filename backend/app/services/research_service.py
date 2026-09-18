import logging

from app.agents.graph import build_research_graph
from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.providers.firecrawl_provider import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider
from app.retrieval.vector_store import ChunkVectorStore
from app.schemas.answer import Answer, AnswerEvidence
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 50  


class ResearchService:
    def __init__(
        self,
        firecrawl: FirecrawlProvider | None = None,
        gemini: GeminiProvider | None = None,
        vector_store: ChunkVectorStore | None = None,
    ) -> None:
        self._graph = build_research_graph(firecrawl, gemini, vector_store)

    def answer(self, question: str) -> Answer:
        try:
            result = self._graph.invoke(
                {
                    "question": question,
                    "search_query": question,
                    "search_results": [],
                    "chunks": [],
                    "selected_chunks": [],
                    "summary": "",
                    "claims": [],
                    "conclusion": "",
                    "retry_count": 0,
                    "evidence_sufficient": False,
                },
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except UpstreamServiceError:
            raise
        except Exception as exc:
            logger.error("Research pipeline failed for question=%r: %s", question, exc)
            raise ResearchServiceError(f"Research pipeline failed: {exc}") from exc

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

        return Answer(
            question=question,
            summary=result["summary"],
            claims=claims,
            conclusion=result["conclusion"],
            evidence=evidence,
            evidence_sufficient=result.get("evidence_sufficient", False),
        )
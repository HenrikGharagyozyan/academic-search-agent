import logging
from collections.abc import Generator
from typing import Any

from app.agents.constants import STAGE_LABELS
from app.agents.graph import build_research_graph
from app.core.exceptions import ResearchServiceError, UpstreamServiceError
from app.infrastructure.search.firecrawl import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider
from app.infrastructure.vector_store.chroma import ChromaVectorStore
from app.domain.answers import Answer, AnswerEvidence
from app.domain.documents import Chunk

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 50


class ResearchService:
    def __init__(
        self,
        firecrawl: FirecrawlProvider | None = None,
        gemini: GeminiProvider | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self._graph = build_research_graph(firecrawl, gemini, vector_store)

    @staticmethod
    def _initial_state(question: str) -> dict[str, Any]:
        return {
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
        }

    def answer(self, question: str) -> Answer:
        try:
            result = self._graph.invoke(
                self._initial_state(question),
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except UpstreamServiceError:
            raise
        except Exception as exc:
            logger.error("Research pipeline failed for question=%r: %s", question, exc, exc_info=True)
            raise ResearchServiceError(f"Research pipeline failed: {exc}") from exc

        return self._build_answer(question, result)

    def stream_answer(self, question: str) -> Generator[dict[str, Any], None, None]:
        """Yields SSE-ready events ({"event": ..., "data": ...}): one "progress"
        event per completed graph node, then a final "result" event with the
        full Answer, or an "error" event if the pipeline fails."""
        state = self._initial_state(question)

        try:
            for update in self._graph.stream(
                state,
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode="updates",
            ):
                for node_name, node_output in update.items():
                    if node_output is None:
                        continue
                    state.update(node_output)
                    yield {
                        "event": "progress",
                        "data": {
                            "stage": node_name,
                            "label": STAGE_LABELS.get(node_name, node_name),
                        },
                    }

            answer = self._build_answer(question, state)
        except UpstreamServiceError as exc:
            yield {"event": "error", "data": {"detail": str(exc)}}
            return
        except Exception as exc:
            logger.error("Streaming research pipeline failed for question=%r: %s", question, exc, exc_info=True)
            yield {"event": "error", "data": {"detail": "Research pipeline failed"}}
            return

        yield {"event": "result", "data": answer.model_dump(mode="json")}

    def _build_answer(self, question: str, result: dict[str, Any]) -> Answer:
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
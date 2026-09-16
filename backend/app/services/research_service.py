import logging

from app.providers.firecrawl_provider import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider
from app.retrieval.chunker import chunk_lines
from app.retrieval.text_splitter import split_into_lines
from app.schemas.answer import Answer, AnswerEvidence
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        firecrawl: FirecrawlProvider | None = None,
        gemini: GeminiProvider | None = None,
        max_sources: int = 3,
    ) -> None:
        self._firecrawl = firecrawl or FirecrawlProvider()
        self._gemini = gemini or GeminiProvider()
        self._max_sources = max_sources

    def answer(self, question: str) -> Answer:
        search_results = self._firecrawl.search(question, limit=self._max_sources)

        all_chunks: list[Chunk] = []
        for result in search_results:
            try:
                page = self._firecrawl.scrape(result.url)
            except Exception:
                logger.warning("Failed to scrape %s, skipping", result.url, exc_info=True)
                continue

            lines = split_into_lines(page.markdown)
            chunks = chunk_lines(
                document_id=result.url,
                lines=lines,
                source_url=page.url,
                title=page.title or result.title,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return Answer(question=question, claims=[], evidence={})

        evidence_chunks = [
            {"chunk_id": c.chunk_id, "text": c.text} for c in all_chunks
        ]

        claims = self._gemini.generate_claims(question, evidence_chunks)

        chunks_by_id = {c.chunk_id: c for c in all_chunks}
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
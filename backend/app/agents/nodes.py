import logging

from app.agents.state import ResearchState
from app.providers.firecrawl_provider import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider
from app.retrieval.chunker import chunk_lines
from app.retrieval.text_splitter import split_into_lines
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)

MAX_SOURCES = 3


def search_node(state: ResearchState, firecrawl: FirecrawlProvider) -> dict:
    results = firecrawl.search(state["question"], limit=MAX_SOURCES)
    return {"search_results": results}


def retrieve_and_chunk_node(state: ResearchState, firecrawl: FirecrawlProvider) -> dict:
    all_chunks: list[Chunk] = []

    for result in state["search_results"]:
        try:
            page = firecrawl.scrape(result.url)
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

    return {"chunks": all_chunks}


def generate_claims_node(state: ResearchState, gemini: GeminiProvider) -> dict:
    if not state["chunks"]:
        return {"claims": []}

    evidence_chunks = [
        {"chunk_id": c.chunk_id, "text": c.text} for c in state["chunks"]
    ]
    claims = gemini.generate_claims(state["question"], evidence_chunks)

    return {"claims": claims}
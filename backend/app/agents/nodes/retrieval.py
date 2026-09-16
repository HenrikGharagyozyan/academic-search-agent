import logging

from app.agents.state import ResearchState
from app.providers.firecrawl_provider import FirecrawlProvider
from app.retrieval.chunker import chunk_lines
from app.retrieval.text_splitter import split_into_lines
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)


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
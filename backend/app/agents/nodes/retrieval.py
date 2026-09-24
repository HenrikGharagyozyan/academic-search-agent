import logging
from concurrent.futures import ThreadPoolExecutor

from app.agents.state import ResearchState
from app.domain.search import ScrapedPage, SearchResult
from app.providers.firecrawl_provider import FirecrawlProvider
from app.retrieval.chunker import chunk_lines, deduplicate_chunks
from app.retrieval.text_splitter import split_into_lines
from app.domain.documents import Chunk
from app.agents.constants import MAX_SCRAPE_WORKERS

logger = logging.getLogger(__name__)



def _scrape_result(
    firecrawl: FirecrawlProvider, result: SearchResult
) -> tuple[SearchResult, ScrapedPage] | None:
    try:
        page = firecrawl.scrape(result.url)
    except Exception:
        logger.warning("Failed to scrape %s, skipping", result.url, exc_info=True)
        return None
    return result, page


def retrieve_and_chunk_node(state: ResearchState, firecrawl: FirecrawlProvider) -> dict:
    search_results = state["search_results"]
    all_chunks: list[Chunk] = []

    if not search_results:
        return {"chunks": all_chunks}

    max_workers = min(len(search_results), MAX_SCRAPE_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = executor.map(
            lambda result: _scrape_result(firecrawl, result), search_results
        )
        for outcome in outcomes:
            if outcome is None:
                continue

            result, page = outcome
            lines = split_into_lines(page.markdown)
            chunks = chunk_lines(
                document_id=result.url,
                lines=lines,
                source_url=page.url,
                title=page.title or result.title,
            )
            all_chunks.extend(chunks)

    return {"chunks": deduplicate_chunks(all_chunks)}
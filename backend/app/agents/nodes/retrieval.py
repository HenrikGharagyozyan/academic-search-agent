import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.state import ResearchState
from app.providers.firecrawl_provider import FirecrawlProvider, ScrapedPage, SearchResult
from app.retrieval.chunker import chunk_lines, deduplicate_chunks
from app.retrieval.text_splitter import split_into_lines
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)

MAX_SCRAPE_WORKERS = 6


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
        futures = [
            executor.submit(_scrape_result, firecrawl, result)
            for result in search_results
        ]
        for future in as_completed(futures):
            outcome = future.result()
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
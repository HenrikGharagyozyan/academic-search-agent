import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.application.agents.activity import ActivityRecorder, count, short_host
from app.application.agents.constants import MAX_SCRAPE_WORKERS
from app.application.agents.state import ResearchState
from app.domain.documents import Chunk
from app.domain.search import SearchResult
from app.domain.text.chunker import chunk_lines, deduplicate_chunks
from app.domain.text.splitter import split_into_lines
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


def _scrape_and_chunk(
    search_provider: SearchProvider, result: SearchResult
) -> list[Chunk] | None:
    """Scrapes one page and cuts it into chunks, or None if it cannot be read.

    Chunking happens here rather than in the caller so that the slow part and
    the CPU part both run off the node's thread, leaving it free to report.
    """
    try:
        page = search_provider.scrape(result.url)
    except Exception:
        logger.warning("Failed to scrape %s, skipping", result.url, exc_info=True)
        return None

    return chunk_lines(
        document_id=result.url,
        lines=split_into_lines(page.markdown),
        source_url=page.url,
        title=page.title or result.title,
    )


def retrieve_and_chunk_node(state: ResearchState, search_provider: SearchProvider) -> dict:
    search_results = state["search_results"]
    if not search_results:
        return {"chunks": []}

    recorder = ActivityRecorder(attempt=state.get("retry_count", 0))
    chunks_by_source: dict[int, list[Chunk]] = {}

    max_workers = min(len(search_results), MAX_SCRAPE_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(_scrape_and_chunk, search_provider, result): (index, result)
            for index, result in enumerate(search_results)
        }

        # as_completed, not executor.map: a step has to be recorded the moment a
        # page lands, and only this thread is allowed to touch the stream writer.
        # map would also hold each result until its predecessors were done.
        for future in as_completed(pending):
            index, result = pending[future]
            chunks = future.result()

            if chunks is None:
                recorder.record(
                    "scrape_failed",
                    f"Could not read {short_host(result.url)}",
                    url=result.url,
                    title=result.title or short_host(result.url),
                )
                continue

            chunks_by_source[index] = chunks
            recorder.record(
                "scrape_ok",
                f"Read {short_host(result.url)}",
                url=result.url,
                title=result.title or short_host(result.url),
                detail=count(len(chunks), "passage"),
            )

    # Flattened in search-result order rather than completion order: which pages
    # finish first is a race, and the vector store falls back to the first
    # top_k chunks when embedding fails, so the order decides what survives.
    ordered = [chunk for index in sorted(chunks_by_source) for chunk in chunks_by_source[index]]
    deduped = deduplicate_chunks(ordered)

    dropped = len(ordered) - len(deduped)
    recorder.record(
        "collect",
        f"Collected {count(len(deduped), 'passage')} from {count(len(chunks_by_source), 'page')}",
        detail=f"{count(dropped, 'duplicate')} dropped" if dropped else None,
    )

    return {"chunks": deduped, "activity": recorder.steps}

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.application.agents.activity import ActivityRecorder, count, short_host
from app.application.agents.constants import MAX_SCRAPE_WORKERS
from app.application.agents.state import ResearchState
from app.domain.documents import Chunk
from app.domain.search import SearchResult
from app.domain.text.chunker import chunk_lines, deduplicate_chunks
from app.domain.text.maths import collapse_wiki_maths
from app.domain.text.plain import to_label
from app.domain.text.similarity import find_mirrors
from app.domain.text.splitter import split_into_lines
from app.infrastructure.search.firecrawl import ScrapeRateLimited
from app.ports.search import SearchProvider

logger = logging.getLogger(__name__)


def _scrape_and_chunk(
    search_provider: SearchProvider, result: SearchResult
) -> list[Chunk] | None | ScrapeRateLimited:
    """Scrapes one page and cuts it into chunks, or None if it cannot be read.

    Chunking happens here rather than in the caller so that the slow part and
    the CPU part both run off the node's thread, leaving it free to report.
    """
    try:
        page = search_provider.scrape(result.url)
    except ScrapeRateLimited as exc:
        # Reported apart from an unreadable page: the reader should know a source
        # was skipped for quota rather than believing the site was broken.
        logger.warning("Rate limited scraping %s", result.url)
        return exc
    except Exception:
        logger.warning("Failed to scrape %s, skipping", result.url, exc_info=True)
        return None

    return chunk_lines(
        document_id=result.url,
        # Before the page is cut up: a formula split across two chunks can no
        # longer be recognised as one.
        lines=split_into_lines(collapse_wiki_maths(page.markdown)),
        source_url=page.url,
        title=page.title or result.title,
        citation=page.citation,
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

            if isinstance(chunks, ScrapeRateLimited):
                recorder.record(
                    "scrape_failed",
                    f"Skipped {short_host(result.url)} — search provider rate limit",
                    url=result.url,
                    title=to_label(result.title) or short_host(result.url),
                    detail="not a problem with the page",
                )
                continue

            if chunks is None:
                recorder.record(
                    "scrape_failed",
                    f"Could not read {short_host(result.url)}",
                    url=result.url,
                    title=to_label(result.title) or short_host(result.url),
                )
                continue

            chunks_by_source[index] = chunks
            recorder.record(
                "scrape_ok",
                f"Read {short_host(result.url)}",
                url=result.url,
                title=to_label(result.title) or short_host(result.url),
                detail=count(len(chunks), "passage"),
            )

    # Pages that turned out to be the same document under different domains are
    # dropped before anything is embedded: six results for one paper is six
    # results, and its passages would otherwise outvote the rest of the field.
    read = sorted(chunks_by_source)
    texts = ["\n".join(c.text for c in chunks_by_source[i]) for i in read]
    # to_label first: arXiv reports its whole metadata table as the title, and
    # two mirrors only match on the part of it that is actually the title.
    titles = [to_label(search_results[i].title) for i in read]
    mirrors = find_mirrors(texts, titles=titles)

    for position, canonical in mirrors.items():
        mirrored, original = read[position], read[canonical]
        recorder.record(
            "mirror_dropped",
            f"{short_host(search_results[mirrored].url)} is the same document as "
            f"{short_host(search_results[original].url)}",
            url=search_results[mirrored].url,
            title=to_label(search_results[mirrored].title) or None,
            detail="kept the fuller version",
        )

    kept = [i for position, i in enumerate(read) if position not in mirrors]

    # Flattened in search-result order rather than completion order: which pages
    # finish first is a race, and the vector store falls back to the first
    # top_k chunks when embedding fails, so the order decides what survives.
    ordered = [chunk for index in kept for chunk in chunks_by_source[index]]
    deduped = deduplicate_chunks(ordered)

    details = []
    if mirrors:
        details.append(f"{count(len(mirrors), 'mirrored page')} dropped")
    if len(ordered) - len(deduped):
        details.append(f"{count(len(ordered) - len(deduped), 'duplicate passage')} dropped")

    recorder.record(
        "collect",
        f"Collected {count(len(deduped), 'passage')} from {count(len(kept), 'distinct page')}",
        detail=", ".join(details) or None,
    )

    return {"chunks": deduped, "activity": recorder.steps}

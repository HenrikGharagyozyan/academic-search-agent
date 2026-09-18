import re
import uuid

from app.schemas.document import Chunk, DocumentLine

MAX_LINES_PER_CHUNK = 20
CHUNK_OVERLAP_LINES = 3


def chunk_lines(
    document_id: str,
    lines: list[DocumentLine],
    source_url: str,
    title: str,
) -> list[Chunk]:
    if not lines:
        return []

    paragraphs: list[list[DocumentLine]] = []
    current: list[DocumentLine] = []

    for i, line in enumerate(lines):
        current.append(line)

        is_last = i == len(lines) - 1
        next_line_number = lines[i + 1].line_number if not is_last else None
        is_gap = next_line_number is not None and next_line_number != line.line_number + 1

        if is_gap or is_last:
            paragraphs.append(current)
            current = []

    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        for piece in _split_with_overlap(paragraph, MAX_LINES_PER_CHUNK, CHUNK_OVERLAP_LINES):
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text="\n".join(l.text for l in piece),
                    start_line=piece[0].line_number,
                    end_line=piece[-1].line_number,
                    source_url=source_url,
                    title=title,
                )
            )

    return chunks


def _split_with_overlap(
    paragraph: list[DocumentLine], max_lines: int, overlap: int
) -> list[list[DocumentLine]]:
    if len(paragraph) <= max_lines:
        return [paragraph]

    step = max(max_lines - overlap, 1)
    pieces: list[list[DocumentLine]] = []
    i = 0
    n = len(paragraph)

    while i < n:
        piece = paragraph[i : i + max_lines]
        pieces.append(piece)
        if i + max_lines >= n:
            break
        i += step

    return pieces


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose text is identical (after whitespace/case normalization)
    to one already seen — catches repeated boilerplate (nav, footers, cookie
    notices) that shows up across multiple scraped pages."""
    seen: set[str] = set()
    deduped: list[Chunk] = []

    for chunk in chunks:
        key = _normalize_for_dedup(chunk.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    return deduped
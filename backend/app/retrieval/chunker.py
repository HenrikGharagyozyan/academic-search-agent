import uuid

from app.schemas.document import Chunk, DocumentLine

MAX_LINES_PER_CHUNK = 20


def chunk_lines(document_id: str, lines: list[DocumentLine]) -> list[Chunk]:
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
        for i in range(0, len(paragraph), MAX_LINES_PER_CHUNK):
            piece = paragraph[i : i + MAX_LINES_PER_CHUNK]
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text="\n".join(l.text for l in piece),
                    start_line=piece[0].line_number,
                    end_line=piece[-1].line_number,
                )
            )

    return chunks
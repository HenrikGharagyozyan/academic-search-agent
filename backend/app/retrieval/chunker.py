import re
import uuid

from app.domain.documents import Chunk, DocumentLine

# Chunks are packed according to the budget of characters, not strings — Firecrawl puts one
# a paragraph (or headline) per line separated by blank lines, so that
# the row limit (MAX_LINES_PER_CHUNK) has never been triggered:
# each "paragraph" is 1 line, and each chunk turned out to be one sentence
# or a bare headline. Packing by characters gives each chunk enough
# the context to be a useful quotable unit, and forces
# MAX_SOURCES/TOP_K_CHUNKS work as intended — a fixed number
# meaningful passages, not a fixed number of random lines.
CHUNK_CHAR_BUDGET = 1400
CHUNK_CHAR_OVERLAP = 200

_HEADING_RE = re.compile(r"^#{1,6}\s")


def chunk_lines(
    document_id: str,
    lines: list[DocumentLine],
    source_url: str,
    title: str,
) -> list[Chunk]:
    if not lines:
        return []

    paragraphs = _group_into_paragraphs(lines)
    paragraphs = _merge_headings_forward(paragraphs)

    chunks: list[Chunk] = []
    for piece in _pack_by_char_budget(paragraphs, CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP):
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=piece.text,
                start_line=piece.start_line,
                end_line=piece.end_line,
                source_url=source_url,
                title=title,
            )
        )

    return chunks


class _Paragraph:
    __slots__ = ("lines",)

    def __init__(self, lines: list[DocumentLine]) -> None:
        self.lines = lines

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def start_line(self) -> int:
        return self.lines[0].line_number

    @property
    def end_line(self) -> int:
        return self.lines[-1].line_number


def _group_into_paragraphs(lines: list[DocumentLine]) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current: list[DocumentLine] = []

    for i, line in enumerate(lines):
        current.append(line)

        is_last = i == len(lines) - 1
        next_line_number = lines[i + 1].line_number if not is_last else None
        is_gap = next_line_number is not None and next_line_number != line.line_number + 1

        if is_gap or is_last:
            paragraphs.append(_Paragraph(current))
            current = []

    return paragraphs


def _merge_headings_forward(paragraphs: list[_Paragraph]) -> list[_Paragraph]:
    """A heading on its own is a useless, orphaned chunk — attach it to the
    paragraph that follows it so it always ships with the content it titles."""
    merged: list[_Paragraph] = []
    pending_heading: _Paragraph | None = None

    for paragraph in paragraphs:
        is_heading = len(paragraph.lines) == 1 and _HEADING_RE.match(paragraph.lines[0].text)

        if is_heading:
            if pending_heading is not None:
                merged.append(pending_heading)
            pending_heading = paragraph
            continue

        if pending_heading is not None:
            merged.append(_Paragraph(pending_heading.lines + paragraph.lines))
            pending_heading = None
        else:
            merged.append(paragraph)

    if pending_heading is not None:
        merged.append(pending_heading)

    return merged


class _Piece:
    __slots__ = ("text", "start_line", "end_line")

    def __init__(self, text: str, start_line: int, end_line: int) -> None:
        self.text = text
        self.start_line = start_line
        self.end_line = end_line


def _split_text(text: str, budget: int) -> list[str]:
    """Cuts a single over-long string into budget-sized parts on word boundaries."""
    parts: list[str] = []
    remaining = text

    while len(remaining) > budget:
        cut = remaining.rfind(" ", 0, budget + 1)
        if cut <= 0:
            cut = budget  # one unbroken run of characters: cut it hard
        head = remaining[:cut].strip()
        if head:
            parts.append(head)
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_oversized(paragraph: _Paragraph, budget: int) -> list[_Paragraph]:
    """Breaks up a paragraph that on its own exceeds the budget.

    A scraped line can be enormous — a whole article rendered without blank
    lines, or a wide table row. Left whole it becomes a chunk many times the
    budget, which pushes the embedding call past the model's input limit; and a
    failed embedding call makes the vector store fall back to "first N chunks"
    for the *entire* request, so one bad page would silently disable semantic
    retrieval for the whole question.
    """
    if len(paragraph.text) <= budget:
        return [paragraph]

    parts: list[_Paragraph] = []
    for line in paragraph.lines:
        for text in _split_text(line.text, budget):
            parts.append(
                _Paragraph([DocumentLine(line_number=line.line_number, text=text)])
            )

    return parts


def _pack_by_char_budget(
    paragraphs: list[_Paragraph], budget: int, overlap: int
) -> list[_Piece]:
    pieces: list[_Piece] = []
    buffer: list[_Paragraph] = []
    carried = 0  # leading paragraphs in the buffer held over from the last piece

    def buffered_chars() -> int:
        """Length of the text flush() would emit, separators included."""
        if not buffer:
            return 0
        return sum(len(p.text) for p in buffer) + 2 * (len(buffer) - 1)

    def flush() -> None:
        nonlocal buffer, carried
        pieces.append(
            _Piece(
                "\n\n".join(p.text for p in buffer),
                buffer[0].start_line,
                buffer[-1].end_line,
            )
        )

        # Carry a tail of the emitted piece so the next one overlaps it. Only
        # paragraphs after the first are eligible: carrying a lone paragraph
        # would emit that same text again as a chunk of its own.
        carry: list[_Paragraph] = []
        carry_chars = 0
        for paragraph in reversed(buffer[1:]):
            if carry and carry_chars >= overlap:
                break
            carry.insert(0, paragraph)
            carry_chars += len(paragraph.text)

        buffer = carry
        carried = len(carry)

    for paragraph in paragraphs:
        for part in _split_oversized(paragraph, budget):
            over_budget = buffered_chars() + 2 + len(part.text) > budget
            # A buffer holding nothing but carry-over has no new content to
            # emit; flushing it would just repeat the previous piece.
            if over_budget and len(buffer) > carried:
                flush()

            buffer.append(part)

    if buffer and len(buffer) > carried:
        pieces.append(
            _Piece(
                "\n\n".join(p.text for p in buffer),
                buffer[0].start_line,
                buffer[-1].end_line,
            )
        )

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
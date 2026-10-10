import re
import uuid

from app.domain.documents import Chunk, DocumentLine
from app.domain.citation import SourceCitation
from app.domain.text.sentences import split_sentences

# Chunks are packed according to the budget of characters, not strings — Firecrawl puts one
# a paragraph (or headline) per line separated by blank lines, so that
# the row limit (MAX_LINES_PER_CHUNK) has never been triggered:
# each "paragraph" is 1 line, and each chunk turned out to be one sentence
# or a bare headline. Packing by characters gives each chunk enough
# the context to be a useful quotable unit, and forces
# MAX_SOURCES/TOP_K_CHUNKS work as intended — a fixed number
# meaningful passages, not a fixed number of random lines.
#
# The budget is a ceiling, not a cutting length. Text is split recursively —
# paragraph, then line, then sentence, then word — and a chunk ends on the
# largest boundary that keeps it within the budget, so a passage never stops in
# the middle of a sentence unless that one sentence is longer than a chunk.
CHUNK_CHAR_BUDGET = 1400
# The next chunk opens with the last whole sentences of this one, up to this
# many characters, so a statement and the sentence explaining it are found
# together whichever chunk is retrieved.
CHUNK_CHAR_OVERLAP = 200

_HEADING_RE = re.compile(r"^#{1,6}\s")


def chunk_lines(
    document_id: str,
    lines: list[DocumentLine],
    source_url: str,
    title: str,
    citation: SourceCitation | None = None,
) -> list[Chunk]:
    if not lines:
        return []

    paragraphs = _group_into_paragraphs(lines)
    paragraphs = _merge_headings_forward(paragraphs)
    units = _to_units(paragraphs, CHUNK_CHAR_BUDGET)

    chunks: list[Chunk] = []
    for piece in _pack(units, CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP):
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=piece.text,
                start_line=piece.start_line,
                end_line=piece.end_line,
                source_url=source_url,
                title=title,
                citation=citation or SourceCitation(),
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


# How a unit joins the one before it in the page: within a line, across a
# line break, or across a blank line.
_SAME_LINE, _NEW_LINE, _NEW_PARAGRAPH = " ", "\n", "\n\n"


class _Unit:
    """The smallest piece a chunk is built from: a sentence, or part of one
    when a sentence alone is longer than the budget."""

    __slots__ = ("text", "line_number", "paragraph", "joiner", "is_heading")

    def __init__(
        self, text: str, line_number: int, paragraph: int, joiner: str, is_heading: bool
    ) -> None:
        self.text = text
        self.line_number = line_number
        self.paragraph = paragraph
        self.joiner = joiner
        self.is_heading = is_heading


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


def _tail(text: str, size: int) -> str:
    """The last ``size`` characters of ``text`` or fewer, starting on a word."""
    if len(text) <= size:
        return text
    start = len(text) - size
    space = text.find(" ", start)
    return text[space + 1 :] if 0 <= space < len(text) - 1 else text[start:]


def _to_units(paragraphs: list[_Paragraph], budget: int) -> list[_Unit]:
    """Paragraph → line → sentence → word: each level is cut only where the
    one above it does not fit, so a cut lands on the largest boundary that
    works.

    A scraped line can be enormous — a whole article rendered without blank
    lines, or a wide table row. Left whole it becomes a chunk many times the
    budget, which pushes the embedding call past the model's input limit; and a
    failed embedding call makes the vector store fall back to ranking by words
    for the *entire* request, so one bad page would disable semantic retrieval
    for the whole question.
    """
    units: list[_Unit] = []
    for index, paragraph in enumerate(paragraphs):
        for line_index, line in enumerate(paragraph.lines):
            is_heading = bool(_HEADING_RE.match(line.text))
            sentences = [line.text] if is_heading else split_sentences(line.text)
            parts = [part for sentence in sentences for part in _split_text(sentence, budget)]
            for part_index, part in enumerate(parts):
                if part_index:
                    joiner = _SAME_LINE
                elif line_index:
                    joiner = _NEW_LINE
                else:
                    joiner = _NEW_PARAGRAPH
                units.append(_Unit(part, line.line_number, index, joiner, is_heading))
    return units


def _join(units: list[_Unit]) -> str:
    return "".join((u.joiner if i else "") + u.text for i, u in enumerate(units))


def _length(units: list[_Unit]) -> int:
    return len(_join(units))


def _pack(units: list[_Unit], budget: int, overlap: int) -> list[_Piece]:
    """Packs units into pieces of at most ``budget`` characters, each opening
    with the last sentences of the one before, up to ``overlap`` characters.

    A paragraph that does not fit is moved whole to the next piece when the
    current one is already half full; otherwise it is cut between sentences, so
    no piece is left mostly empty to keep a paragraph together.
    """
    pieces: list[_Piece] = []
    buffer: list[_Unit] = []
    carried = 0  # leading units in the buffer held over from the last piece

    def emit(content: list[_Unit]) -> None:
        pieces.append(
            _Piece(_join(content), content[0].line_number, content[-1].line_number)
        )

    def overlap_from(content: list[_Unit]) -> list[_Unit]:
        """The trailing sentences to repeat at the start of the next piece.

        Never the whole piece: a piece repeated in full is the same passage
        twice, embedded twice and competing with itself for the top slots.
        When even the last sentence is longer than the overlap, its tail is
        carried instead, cut to start on a word.
        """
        carry: list[_Unit] = []
        for unit in reversed(content[1:]):
            if _length([unit] + carry) > overlap:
                break
            carry.insert(0, unit)
        if carry or len(_join(content)) <= overlap:
            return carry
        last = content[-1]
        fragment = _tail(last.text, overlap)
        return [_Unit(fragment, last.line_number, last.paragraph, last.joiner, False)]

    def flush() -> None:
        nonlocal buffer, carried
        content = buffer
        # A heading closing a piece would be cut off from what it titles.
        held = []
        while len(content) > carried + 1 and content[-1].is_heading:
            held.insert(0, content.pop())
        emit(content)
        buffer = overlap_from(content) + held
        carried = len(buffer) - len(held)

    paragraph_lengths: dict[int, int] = {}
    for unit in units:
        paragraph_lengths[unit.paragraph] = paragraph_lengths.get(unit.paragraph, 0) + len(
            unit.joiner
        ) + len(unit.text)

    for unit in units:
        fresh = len(buffer) > carried
        if fresh and unit.joiner == _NEW_PARAGRAPH:
            room = budget - _length(buffer)
            if paragraph_lengths[unit.paragraph] > room and _length(buffer) >= budget // 2:
                flush()

        while _length(buffer + [unit]) > budget:
            if len(buffer) > carried:
                flush()
            elif buffer:
                # Only overlap left, and the unit does not fit beside it:
                # give up overlap before going over the budget.
                buffer.pop(0)
                carried -= 1
            else:
                break

        buffer.append(unit)

    if len(buffer) > carried:
        emit(buffer)

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
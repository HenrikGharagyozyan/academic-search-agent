"""Documents as the pipeline sees them: a scraped page, its numbered lines,
and the chunks those lines are packed into."""

from pydantic import BaseModel


class DocumentLine(BaseModel):
    line_number: int
    text: str


class ParsedDocument(BaseModel):
    url: str
    title: str
    lines: list[DocumentLine]


class Chunk(BaseModel):
    """A quotable passage: the unit that gets embedded, graded and cited."""

    chunk_id: str
    document_id: str
    text: str
    start_line: int
    end_line: int
    source_url: str
    title: str

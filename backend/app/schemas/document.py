from pydantic import BaseModel


class DocumentRequest(BaseModel):
    url: str


class DocumentLine(BaseModel):
    line_number: int
    text: str


class ParsedDocument(BaseModel):
    url: str
    title: str
    lines: list[DocumentLine]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    start_line: int
    end_line: int
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
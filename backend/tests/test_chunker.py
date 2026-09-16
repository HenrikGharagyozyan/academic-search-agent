from app.retrieval.chunker import chunk_lines
from app.schemas.document import DocumentLine


def test_chunk_lines_groups_by_paragraph():
    lines = [
        DocumentLine(line_number=1, text="First paragraph line one"),
        DocumentLine(line_number=2, text="First paragraph line two"),
        DocumentLine(line_number=5, text="Second paragraph line one"),
    ]

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 5


def test_chunk_lines_empty():
    assert chunk_lines("doc1", [], source_url="https://example.com", title="Example") == []


def test_chunk_lines_splits_oversized_paragraph():
    lines = [DocumentLine(line_number=i, text=f"line {i}") for i in range(1, 45)]

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    assert len(chunks) == 3
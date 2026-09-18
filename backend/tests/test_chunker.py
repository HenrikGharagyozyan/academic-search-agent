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


def test_chunk_lines_overlap_between_consecutive_pieces():
    lines = [DocumentLine(line_number=i, text=f"line {i}") for i in range(1, 45)]
    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    assert len(chunks) == 3
    assert chunks[1].start_line <= chunks[0].end_line
    assert chunks[2].start_line <= chunks[1].end_line


def test_deduplicate_chunks_drops_repeated_boilerplate():
    from app.retrieval.chunker import deduplicate_chunks
    from app.schemas.document import Chunk

    def make_chunk(text, source_url):
        return Chunk(
            chunk_id="x", document_id="d", text=text,
            start_line=1, end_line=1, source_url=source_url, title="T",
        )

    chunks = [
        make_chunk("Cookie policy. Accept all.", "https://a.com"),
        make_chunk("Real unique research content here.", "https://a.com"),
        make_chunk("Cookie policy.   Accept all.", "https://b.com"),  # тот же текст с другим пробелом
    ]

    result = deduplicate_chunks(chunks)

    assert len(result) == 2
    assert result[0].source_url == "https://a.com"
    assert result[1].text == "Real unique research content here."
from app.retrieval.chunker import chunk_lines, deduplicate_chunks
from app.retrieval.text_splitter import split_into_lines
from app.schemas.document import Chunk, DocumentLine


def test_chunk_lines_packs_small_paragraphs_into_one_chunk():
    lines = [
        DocumentLine(line_number=1, text="First paragraph line one"),
        DocumentLine(line_number=2, text="First paragraph line two"),
        DocumentLine(line_number=5, text="Second paragraph line one"),
    ]

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    # Both paragraphs are tiny, so they get packed into a single chunk
    # instead of one chunk per paragraph — that's the whole point of the fix.
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[0].text


def test_chunk_lines_empty():
    assert chunk_lines("doc1", [], source_url="https://example.com", title="Example") == []


def test_chunk_lines_splits_when_content_exceeds_budget():
    paragraph_text = "x" * 500
    lines = []
    line_number = 1
    for _ in range(5):
        lines.append(DocumentLine(line_number=line_number, text=paragraph_text))
        line_number += 2  # gap -> paragraph boundary, mirrors a blank-line separator

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    # 5 paragraphs of 500 chars each (2500 total) against a 1400-char budget
    # must produce more than one chunk, but fewer than one-per-paragraph —
    # they get packed together with overlap, not split 1:1.
    assert 1 < len(chunks) < 5


def test_chunk_lines_overlap_between_consecutive_pieces():
    paragraph_text = "x" * 500
    lines = []
    line_number = 1
    for _ in range(5):
        lines.append(DocumentLine(line_number=line_number, text=paragraph_text))
        line_number += 2

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_line <= chunks[i].end_line


def test_chunk_lines_merges_heading_into_following_paragraph():
    lines = [
        DocumentLine(line_number=1, text="## Section 4"),
        DocumentLine(line_number=3, text="Some real paragraph content about the topic."),
    ]

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    # A lone heading is never a useful citation on its own — it must ship
    # together with the content it introduces.
    assert len(chunks) == 1
    assert chunks[0].text.startswith("## Section 4")
    assert "Some real paragraph content" in chunks[0].text


def test_chunk_lines_keeps_trailing_heading_without_following_content():
    lines = [DocumentLine(line_number=1, text="## Trailing heading")]

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    # Edge case: a heading with nothing after it (end of document) must not
    # be silently dropped — it just stays as its own chunk.
    assert len(chunks) == 1
    assert chunks[0].text == "## Trailing heading"


def test_chunk_lines_on_realistic_firecrawl_markdown():
    """Regression test for the original bug: Firecrawl markdown puts one
    paragraph or heading per line, separated by blank lines. The old
    line-count-based chunker turned every single line into its own chunk
    (20 lines -> 20 chunks, headers included). This pins the fix down."""
    paragraph = (
        "This is a realistic paragraph of body text that a real article "
        "would contain, long enough to look like actual prose rather than "
        "a single short line, discussing some topic in reasonable depth."
    )
    markdown_parts = []
    for i in range(1, 11):
        markdown_parts.append(f"## Section {i}")
        markdown_parts.append("")
        markdown_parts.append(paragraph)
        markdown_parts.append("")
    markdown = "\n".join(markdown_parts)

    lines = split_into_lines(markdown)
    non_blank_line_count = len(lines)

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    # Old behaviour: len(chunks) == non_blank_line_count (1:1). New
    # behaviour: chunks pack multiple sections together.
    assert len(chunks) < non_blank_line_count

    # No chunk is a bare, content-free heading.
    single_line_headings = [
        c
        for c in chunks
        if c.text.strip().startswith("#") and "\n" not in c.text.strip() and len(c.text.strip()) < 40
    ]
    assert not single_line_headings

    # Every chunk carries enough content to be a useful citation.
    assert all(len(c.text) > 60 for c in chunks)


def test_deduplicate_chunks_drops_repeated_boilerplate():
    def make_chunk(text, source_url):
        return Chunk(
            chunk_id="x", document_id="d", text=text,
            start_line=1, end_line=1, source_url=source_url, title="T",
        )

    chunks = [
        make_chunk("Cookie policy. Accept all.", "https://a.com"),
        make_chunk("Real unique research content here.", "https://a.com"),
        make_chunk("Cookie policy.   Accept all.", "https://b.com"),  # same text with different spacing
    ]

    result = deduplicate_chunks(chunks)

    assert len(result) == 2
    assert result[0].source_url == "https://a.com"
    assert result[1].text == "Real unique research content here."

def test_chunk_lines_does_not_emit_a_lone_paragraph_twice():
    """The overlap carried a whole paragraph forward even when the buffer held
    only that one paragraph, so it shipped once alone and again with the next
    one — duplicate embeddings, and duplicates competing for the top-k slots."""
    big = "B" * 1300  # fits the budget alone, leaves no room for a neighbour
    markdown = "\n\n".join([big, "S" * 200, "S" * 200, "S" * 200])
    lines = split_into_lines(markdown)

    chunks = chunk_lines("doc1", lines, source_url="https://example.com", title="Example")

    texts = [c.text for c in chunks]
    for i, text in enumerate(texts):
        others = texts[:i] + texts[i + 1 :]
        assert not any(text in other for other in others), "a chunk is contained in another"

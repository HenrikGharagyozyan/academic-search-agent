from app.domain.documents import DocumentLine


def split_into_lines(markdown: str) -> list[DocumentLine]:
    raw_lines = markdown.splitlines()

    return [
        DocumentLine(line_number=i + 1, text=line)
        for i, line in enumerate(raw_lines)
        if line.strip()
    ]
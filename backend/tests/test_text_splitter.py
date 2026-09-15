from app.retrieval.text_splitter import split_into_lines


def test_split_into_lines_skips_empty_lines():
    markdown = "Line one\n\nLine three\n"

    result = split_into_lines(markdown)

    assert len(result) == 2
    assert result[0].line_number == 1
    assert result[0].text == "Line one"
    assert result[1].line_number == 3
    assert result[1].text == "Line three"


def test_split_into_lines_empty_input():
    assert split_into_lines("") == []
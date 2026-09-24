from unittest.mock import MagicMock

from app.agents.nodes.grading import grade_relevance_node, grade_answer_node
from app.domain.documents import Chunk
from app.domain.grading import RelevanceGrade, AnswerQualityGrade
from app.domain.answers import Claim


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", text="text",
        start_line=1, end_line=1, source_url="https://x.com", title="X",
    )


def test_grade_relevance_filters_irrelevant_chunks():
    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.return_value = RelevanceGrade(
        relevant_chunk_ids=["c1"], reasoning="only c1 is on topic"
    )

    state = {"question": "q?", "selected_chunks": [make_chunk("c1"), make_chunk("c2")]}
    result = grade_relevance_node(state, gemini=mock_gemini)

    assert [c.chunk_id for c in result["selected_chunks"]] == ["c1"]


def test_grade_relevance_drops_everything_if_none_relevant():
    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.return_value = RelevanceGrade(
        relevant_chunk_ids=[], reasoning="none relevant"
    )

    chunks = [make_chunk("c1"), make_chunk("c2")]
    state = {"question": "q?", "selected_chunks": chunks}
    result = grade_relevance_node(state, gemini=mock_gemini)

    # An empty answer beats one built on irrelevant chunks.
    assert result == {"selected_chunks": []}


def test_grade_relevance_keeps_all_when_grading_fails():
    mock_gemini = MagicMock()
    mock_gemini.grade_relevance.side_effect = RuntimeError("upstream down")

    chunks = [make_chunk("c1"), make_chunk("c2")]
    state = {"question": "q?", "selected_chunks": chunks}
    result = grade_relevance_node(state, gemini=mock_gemini)

    # A failing judge is no reason to lose context: the state stays untouched.
    assert result == {}


def test_grade_answer_marks_insufficient_when_unsatisfactory():
    mock_gemini = MagicMock()
    mock_gemini.grade_answer_quality.return_value = AnswerQualityGrade(
        is_satisfactory=False, reasoning="off topic"
    )

    state = {
        "question": "q?", "summary": "s", "conclusion": "c",
        "claims": [Claim(text="x", evidence_ids=["c1"], confidence="high")],
    }
    result = grade_answer_node(state, gemini=mock_gemini)

    assert result == {"evidence_sufficient": False}


def test_grade_answer_returns_empty_when_satisfactory():
    mock_gemini = MagicMock()
    mock_gemini.grade_answer_quality.return_value = AnswerQualityGrade(
        is_satisfactory=True, reasoning="good"
    )

    state = {
        "question": "q?", "summary": "s", "conclusion": "c",
        "claims": [Claim(text="x", evidence_ids=["c1"], confidence="high")],
    }
    result = grade_answer_node(state, gemini=mock_gemini)

    assert result == {}
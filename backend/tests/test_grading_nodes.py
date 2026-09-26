from unittest.mock import MagicMock

from app.application.agents.nodes.grading import grade_relevance_node, grade_answer_node
from app.domain.documents import Chunk
from app.domain.grading import RelevanceGrade, AnswerQualityGrade
from app.domain.answers import Claim


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", text="text",
        start_line=1, end_line=1, source_url="https://x.com", title="X",
    )


def test_grade_relevance_filters_irrelevant_chunks():
    mock_llm = MagicMock()
    mock_llm.grade_relevance.return_value = RelevanceGrade(
        relevant_chunk_ids=["c1"], reasoning="only c1 is on topic"
    )

    state = {"question": "q?", "selected_chunks": [make_chunk("c1"), make_chunk("c2")]}
    result = grade_relevance_node(state, llm=mock_llm)

    assert [c.chunk_id for c in result["selected_chunks"]] == ["c1"]


def test_grade_relevance_drops_everything_if_none_relevant():
    mock_llm = MagicMock()
    mock_llm.grade_relevance.return_value = RelevanceGrade(
        relevant_chunk_ids=[], reasoning="none relevant"
    )

    chunks = [make_chunk("c1"), make_chunk("c2")]
    state = {"question": "q?", "selected_chunks": chunks}
    result = grade_relevance_node(state, llm=mock_llm)

    # An empty answer beats one built on irrelevant chunks.
    assert result["selected_chunks"] == []
    assert [step.kind for step in result["activity"]] == ["grade_relevance"]


def test_grade_relevance_keeps_all_when_grading_fails():
    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = RuntimeError("upstream down")

    chunks = [make_chunk("c1"), make_chunk("c2")]
    state = {"question": "q?", "selected_chunks": chunks}
    result = grade_relevance_node(state, llm=mock_llm)

    # A failing judge is no reason to lose context: the chunks are left alone.
    assert "selected_chunks" not in result
    # The reader still gets told the judge was skipped rather than silence.
    assert result["activity"][0].kind == "grade_relevance"
    assert "keeping all" in result["activity"][0].label


def test_grade_answer_marks_insufficient_when_unsatisfactory():
    mock_llm = MagicMock()
    mock_llm.grade_answer_quality.return_value = AnswerQualityGrade(
        is_satisfactory=False, reasoning="off topic"
    )

    state = {
        "question": "q?", "summary": "s", "conclusion": "c",
        "claims": [Claim(text="x", evidence_ids=["c1"], confidence="high")],
    }
    result = grade_answer_node(state, llm=mock_llm)

    assert result["evidence_sufficient"] is False
    assert result["activity"][0].detail == "off topic"


def test_grade_answer_returns_empty_when_satisfactory():
    mock_llm = MagicMock()
    mock_llm.grade_answer_quality.return_value = AnswerQualityGrade(
        is_satisfactory=True, reasoning="good"
    )

    state = {
        "question": "q?", "summary": "s", "conclusion": "c",
        "claims": [Claim(text="x", evidence_ids=["c1"], confidence="high")],
    }
    result = grade_answer_node(state, llm=mock_llm)

    # Nothing about the verdict changes; only the trail grows.
    assert "evidence_sufficient" not in result
    assert result["activity"][0].label.endswith("satisfactory")
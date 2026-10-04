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

    # A failing judge is no reason to lose context: the batch keeps its chunks.
    assert [c.chunk_id for c in result["selected_chunks"]] == ["c1", "c2"]
    # The reader still gets told the judgement was skipped rather than silence.
    assert result["activity"][0].kind == "grade_relevance"
    assert "could not be judged" in result["activity"][0].detail


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


def test_an_insufficient_answer_leaves_its_shortfall_for_the_rewrite():
    mock_llm = MagicMock()
    mock_llm.grade_answer_quality.return_value = AnswerQualityGrade(
        is_satisfactory=False,
        reasoning="covers base editing only",
        problem="missing_aspect",
        missing="how prime editing differs from base editing",
    )
    state = {
        "question": "q?", "summary": "s", "conclusion": "c",
        "claims": [Claim(text="x", evidence_ids=["c1"], confidence="high")],
    }

    shortfall = grade_answer_node(state, llm=mock_llm)["shortfall"]

    assert "missing part of the question" in shortfall
    assert "how prime editing differs from base editing" in shortfall


def test_a_shortfall_without_missing_falls_back_to_the_reasoning():
    from app.application.agents.nodes.grading import describe_shortfall

    grade = AnswerQualityGrade(is_satisfactory=False, reasoning="only one weak source", problem="too_thin")

    assert describe_shortfall(grade) == "The answer was on topic but too thin. It needed: only one weak source"


def test_no_grounded_claims_is_a_shortfall_too():
    from app.application.agents.nodes.grading import NO_GROUNDED_CLAIMS

    result = grade_answer_node({"question": "q?", "claims": []}, llm=MagicMock())

    assert result["shortfall"] == NO_GROUNDED_CLAIMS


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

def test_passages_are_judged_in_batches_not_all_at_once():
    """One call over forty passages kept under a fifth of them; batches of eight
    keep about half. Measured on a fixed set — the prompt was not the lever."""
    from app.application.agents.constants import GRADE_BATCH_SIZE

    chunks = [make_chunk(f"c{i}") for i in range(GRADE_BATCH_SIZE * 3)]
    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = lambda q, batch: RelevanceGrade(
        relevant_chunk_ids=[c.chunk_id for c in batch], reasoning="all"
    )

    result = grade_relevance_node({"question": "q?", "selected_chunks": chunks}, llm=mock_llm)

    assert mock_llm.grade_relevance.call_count == 3
    assert all(
        len(call.args[1]) <= GRADE_BATCH_SIZE
        for call in mock_llm.grade_relevance.call_args_list
    )
    assert len(result["selected_chunks"]) == len(chunks)


def test_one_failed_batch_does_not_discard_the_others_judgement():
    from app.application.agents.constants import GRADE_BATCH_SIZE

    chunks = [make_chunk(f"c{i}") for i in range(GRADE_BATCH_SIZE * 2)]
    calls = {"n": 0}

    def grade(question, batch):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("upstream down")
        # the second judge rejects everything it sees
        return RelevanceGrade(relevant_chunk_ids=[], reasoning="none relevant")

    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = grade

    result = grade_relevance_node({"question": "q?", "selected_chunks": chunks}, llm=mock_llm)
    kept = {c.chunk_id for c in result["selected_chunks"]}

    # The unjudged batch is kept, the judged one honoured. Not all-or-nothing.
    assert len(kept) == GRADE_BATCH_SIZE


def test_the_order_of_the_ranking_is_preserved_across_batches():
    chunks = [make_chunk(f"c{i}") for i in range(12)]
    mock_llm = MagicMock()
    mock_llm.grade_relevance.side_effect = lambda q, batch: RelevanceGrade(
        relevant_chunk_ids=[batch[0].chunk_id], reasoning="first only"
    )

    result = grade_relevance_node({"question": "q?", "selected_chunks": chunks}, llm=mock_llm)
    kept = [c.chunk_id for c in result["selected_chunks"]]

    assert kept == sorted(kept, key=lambda cid: [c.chunk_id for c in chunks].index(cid))

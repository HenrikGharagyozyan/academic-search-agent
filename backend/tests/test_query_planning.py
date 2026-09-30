"""Planning several differently-aimed searches instead of sending one query.

The case this exists for: "latest research on transmission matrix engineering"
returned six results that were one paper mirrored across arXiv, the publisher,
a lab page and two aggregators. Web search ranks by links, so one query on a
well-studied topic spends its whole budget on the most-cited work.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.application.agents.constants import (
    MAX_QUERIES,
    MAX_SOURCES,
    RECENCY_WINDOW_YEARS,
    RESULTS_PER_QUERY,
)
from app.application.agents.nodes import plan_searches_node, search_node
from app.core.exceptions import UpstreamServiceError
from app.domain.query import QueryPlan, anchor_terms, has_recency_intent, stays_on_topic
from app.domain.search import SearchResult

QUESTION = "latest research on transmission matrix engineering"


def result(url: str, title: str = "A paper") -> SearchResult:
    return SearchResult(title=title, url=url, snippet="...")


# --- recency detection -----------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "latest research on transmission matrix engineering",
        "recent advances in wavefront shaping",
        "state-of-the-art fiber endoscopy",
        "what is new in transmission matrices",
        "current trends in multimode fibers",
        "последние исследования по матрице пропускания",
        "недавние работы по волновому фронту",
    ],
)
def test_a_question_about_current_work_is_recognised(question):
    assert has_recency_intent(question, today=date(2026, 9, 27))


@pytest.mark.parametrize(
    "question",
    [
        "what is a transmission matrix",
        "difference between Adam and AdamW",
        "how does gradient descent converge for convex functions",
        "the 1998 proof of Fermat's last theorem",
        "Shannon's 1948 paper on information theory",
        "",
    ],
)
def test_a_timeless_question_is_not_treated_as_recent(question):
    assert not has_recency_intent(question, today=date(2026, 9, 27))


def test_a_recent_year_in_the_question_counts_but_an_old_one_does_not():
    today = date(2026, 9, 27)

    assert has_recency_intent("transmission matrix 2025", today=today)
    assert not has_recency_intent("transmission matrix 1998", today=today)


# --- planning --------------------------------------------------------------


def test_planning_asks_for_several_queries_and_keeps_them():
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(
        queries=[
            QUESTION,
            "online learning transmission matrix multimode fiber 2025",
            "reference-free transmission matrix retrieval",
            "transmission-matrix-free endoscopy deep learning",
        ],
        reasoning="four directions",
    )

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)

    assert len(out["search_queries"]) == MAX_QUERIES
    # The planner was told the question is time-bound.
    assert llm.plan_searches.call_args.kwargs["recent"] is True
    assert out["since_year"] == date.today().year - RECENCY_WINDOW_YEARS


def test_a_timeless_question_gets_no_date_filter():
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(queries=["what is a transmission matrix"])

    out = plan_searches_node(
        {"question": "what is a transmission matrix", "search_query": "what is a transmission matrix"},
        llm=llm,
    )

    assert out["since_year"] is None
    assert llm.plan_searches.call_args.kwargs["recent"] is False


def test_the_question_itself_is_always_searched():
    """A planner that wanders off the topic must not be able to lose it."""
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(queries=["something else entirely"])

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)

    assert out["search_queries"][0] == QUESTION


def test_planning_falls_back_to_the_bare_question_when_the_llm_fails():
    # Degraded to exactly the behaviour this node replaced — never worse.
    llm = MagicMock()
    llm.plan_searches.side_effect = RuntimeError("model unavailable")

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)

    assert out["search_queries"] == [QUESTION]
    assert out["activity"][0].kind == "plan"


def test_blank_queries_from_the_planner_are_discarded():
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(
        queries=["  ", "", "transmission matrix engineering review"]
    )

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)

    assert out["search_queries"] == [QUESTION, "transmission matrix engineering review"]


# --- searching several queries ---------------------------------------------


def test_every_planned_query_is_searched():
    provider = MagicMock()
    provider.search.side_effect = lambda q, limit, since_year: [result(f"https://{q[:3]}.com")]
    state = {"question": QUESTION, "search_queries": ["alpha", "beta", "gamma"], "since_year": None}

    search_node(state, provider)

    assert [c.args[0] for c in provider.search.call_args_list] == ["alpha", "beta", "gamma"]


def test_the_date_filter_reaches_the_provider():
    provider = MagicMock()
    provider.search.return_value = [result("https://a.com")]
    state = {"question": QUESTION, "search_queries": ["alpha"], "since_year": 2023}

    search_node(state, provider)

    assert provider.search.call_args.kwargs["since_year"] == 2023
    assert provider.search.call_args.kwargs["limit"] == RESULTS_PER_QUERY


def test_results_are_taken_round_robin_so_later_queries_are_not_starved():
    """Concatenating would let the first query fill the budget on its own, and
    the queries deliberately aimed elsewhere would contribute nothing."""
    provider = MagicMock()
    per_query = {
        "alpha": [result(f"https://a{i}.com") for i in range(4)],
        "beta": [result(f"https://b{i}.com") for i in range(4)],
        "gamma": [result(f"https://g{i}.com") for i in range(4)],
    }
    provider.search.side_effect = lambda q, limit, since_year: per_query[q]
    state = {
        "question": QUESTION,
        "search_queries": ["alpha", "beta", "gamma"],
        "since_year": None,
    }

    out = search_node(state, provider)
    hosts = [r.url for r in out["search_results"]]

    assert len(hosts) == MAX_SOURCES
    # Every query is represented in the first round.
    assert {hosts[0][:10], hosts[1][:10], hosts[2][:10]} == {
        "https://a0", "https://b0", "https://g0",
    }
    for prefix in ("a", "b", "g"):
        assert any(h.startswith(f"https://{prefix}") for h in hosts)


def test_the_same_url_found_by_two_queries_is_taken_once():
    provider = MagicMock()
    shared = result("https://arxiv.org/abs/1")
    provider.search.side_effect = lambda q, limit, since_year: [shared, result(f"https://{q}.com")]
    state = {"question": QUESTION, "search_queries": ["alpha", "beta"], "since_year": None}

    out = search_node(state, provider)
    urls = [r.url for r in out["search_results"]]

    assert urls.count("https://arxiv.org/abs/1") == 1


def test_one_failing_query_does_not_sink_the_others():
    provider = MagicMock()

    def search(q, limit, since_year):
        if q == "broken":
            raise RuntimeError("upstream hiccup")
        return [result(f"https://{q}.com")]

    provider.search.side_effect = search
    state = {"question": QUESTION, "search_queries": ["broken", "fine"], "since_year": None}

    out = search_node(state, provider)

    assert [r.url for r in out["search_results"]] == ["https://fine.com"]
    assert any("failed" in s.label.lower() for s in out["activity"])


def test_every_query_failing_is_an_outage():
    provider = MagicMock()
    provider.search.side_effect = RuntimeError("provider down")
    state = {"question": QUESTION, "search_queries": ["alpha", "beta"], "since_year": None}

    with pytest.raises(UpstreamServiceError):
        search_node(state, provider)


def test_the_date_filter_is_googles_custom_range_syntax():
    """Firecrawl passes tbs straight through to the engine, so the format is
    Google's. The relative form (qdr:y, past year) is too tight for a
    literature question — the interesting work in the reported case was two
    years old."""
    from app.infrastructure.search.firecrawl import _date_filter

    assert _date_filter(2023) == "cdr:1,cd_min:1/1/2023"


# --- staying inside the subject --------------------------------------------

# Every string below came out of a real run. The first group is what the planner
# produced when it worked; the second is the drift that made the agent answer
# about the nonlinear Schrödinger equation and the Born approximation instead of
# about multimode fibers.
ON_TOPIC = [
    "transmission matrix engineering 2023",
    "metasurface transmission matrix techniques",
    "adaptive optics transmission matrix applications",
    "transmission matrix engineering recent research",
    "real-time transmission matrix characterization",
    "machine learning methods for transmission matrix reconstruction",
]

DRIFTED = [
    "wavefront shaping techniques 2022",
    "adaptive optics for imaging 2023",
    "inverse scattering theory",
    "nonlinear Schrodinger equation solitons",
    "Born approximation weak scattering",
]


def test_the_questions_subject_words_are_what_anchor_it():
    # "latest" and "research" say it is a research question, not what about.
    assert anchor_terms(QUESTION) == {"transmission", "matrix", "engineering"}


@pytest.mark.parametrize("query", ON_TOPIC)
def test_a_query_that_keeps_the_subject_is_searched(query):
    assert stays_on_topic(query, anchor_terms(QUESTION))


@pytest.mark.parametrize("query", DRIFTED)
def test_a_query_that_generalises_out_of_the_subject_is_not(query):
    assert not stays_on_topic(query, anchor_terms(QUESTION))


def test_drifted_queries_never_reach_the_search():
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(queries=[QUESTION, *DRIFTED, *ON_TOPIC[:2]])

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)

    assert out["search_queries"] == [QUESTION, *ON_TOPIC[:2]]


def test_the_trail_says_which_queries_were_discarded():
    """Silently dropping them would make a narrow answer look inexplicable."""
    llm = MagicMock()
    llm.plan_searches.return_value = QueryPlan(queries=[QUESTION, "inverse scattering theory"])

    out = plan_searches_node({"question": QUESTION, "search_query": QUESTION}, llm=llm)
    discarded = [s for s in out["activity"] if "left the subject" in s.label]

    assert len(discarded) == 1
    assert "inverse scattering theory" in discarded[0].detail


def test_a_question_with_no_subject_words_anchors_nothing():
    # Nothing to enforce against, so nothing is discarded: better a broad search
    # than no search.
    anchors = anchor_terms("what is the latest research")

    assert anchors == frozenset()
    assert stays_on_topic("anything at all", anchors)


def test_a_one_word_subject_still_has_to_appear():
    anchors = anchor_terms("recent work on photobleaching")

    assert anchors == {"photobleaching"}
    assert stays_on_topic("photobleaching kinetics 2025", anchors)
    assert not stays_on_topic("fluorescence microscopy advances", anchors)


def test_the_relevance_judge_separates_subject_from_angle():
    """Both failure modes have happened. Judging only the register let
    inverse-scattering chunks through as "on topic"; judging the subject too
    narrowly then threw away 26 of 30 passages, including different methods for
    the same problem, and the answer covered three approaches instead of six."""
    from app.infrastructure.llm.prompts import RELEVANCE_GRADE_SYSTEM_PROMPT as prompt

    assert "SUBJECT versus ANGLE" in prompt
    # A different method for the same problem must be kept...
    assert "Do not reject a chunk for approaching the subject differently" in prompt
    assert "Err towards keeping" in prompt
    # ...while a different subject sharing the vocabulary must not be.
    assert "REJECT inverse scattering theory" in prompt
    assert "adaptive optics in ophthalmology" in prompt
    # And finding nothing is still an acceptable outcome.
    assert "returning nothing is still a" in prompt


def test_the_judge_is_no_longer_told_to_be_strict_for_its_own_sake():
    """These three lines are what over-tightened it."""
    from app.infrastructure.llm.prompts import RELEVANCE_GRADE_SYSTEM_PROMPT as prompt

    assert "Being strict on subject is the point" not in prompt
    assert "Do not stretch to fill a quota" not in prompt
    assert "Not the general area" not in prompt


def test_the_expansion_prompt_shows_what_drift_looks_like():
    from app.infrastructure.llm.prompts import EXPANSION_SYSTEM_PROMPT

    assert "STAY IN THE SUBJECT" in EXPANSION_SYSTEM_PROMPT
    # The examples name the exact failure that was reported.
    assert "BAD  — inverse scattering theory" in EXPANSION_SYSTEM_PROMPT

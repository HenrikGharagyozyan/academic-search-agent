"""Ranking by the question's words, used when embeddings cannot be had."""

from app.domain.text.lexical import rank_by_terms

NAV = "Main page Contents Current events Random article About Wikipedia Donate"
INTRO = "This article is about the concept in physics. For other uses, see Energy."
BOLTZMANN = (
    "In statistical mechanics the entropy of a system is proportional to the "
    "logarithm of its number of microstates, S = k_B ln Ω."
)
HISTORY = "Clausius coined the word entropy in 1865 while studying heat engines."


def test_a_passage_about_the_subject_outranks_the_top_of_the_page():
    order = rank_by_terms(
        "What is entropy in statistical mechanics?", [NAV, INTRO, HISTORY, BOLTZMANN]
    )

    assert order[0] == 3
    assert order[1] == 2


def test_every_passage_is_returned_and_the_unmatched_keep_page_order():
    order = rank_by_terms("What is entropy?", [NAV, BOLTZMANN, INTRO])

    assert order == [1, 0, 2]


def test_a_plural_in_the_question_matches_the_singular_in_the_passage():
    order = rank_by_terms(
        "neural radiance fields", ["Cooking with gas.", "A radiance field stores density."]
    )

    assert order[0] == 1


def test_a_long_passage_does_not_win_by_length_alone():
    padded = "entropy " + "filler word " * 400
    focused = "Entropy measures the number of microstates of a system."

    assert rank_by_terms("entropy microstates", [padded, focused])[0] == 1


def test_a_question_with_no_subject_words_leaves_the_order_alone():
    assert rank_by_terms("what is the latest research?", ["a", "b", "c"]) == [0, 1, 2]


def test_no_passages_is_not_an_error():
    assert rank_by_terms("entropy", []) == []

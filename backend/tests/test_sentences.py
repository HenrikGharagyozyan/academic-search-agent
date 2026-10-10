from app.domain.text.sentences import split_sentences


def test_splits_on_end_punctuation():
    assert split_sentences("One is here. Two is here! Three? Four.") == [
        "One is here.",
        "Two is here!",
        "Three?",
        "Four.",
    ]


def test_keeps_scientific_abbreviations_inside_the_sentence():
    text = "Cheng et al. (2023) report it in Fig. 3 and Eq. 2. The next result differs."
    assert split_sentences(text) == [
        "Cheng et al. (2023) report it in Fig. 3 and Eq. 2.",
        "The next result differs.",
    ]


def test_an_initial_does_not_end_the_sentence():
    assert split_sentences("Written by A. Smith in Nature. 2025 followed.") == [
        "Written by A. Smith in Nature.",
        "2025 followed.",
    ]


def test_a_full_stop_inside_maths_does_not_end_the_sentence():
    assert split_sentences("We set $x = 1. Y$ in the model. Then it converges.") == [
        "We set $x = 1. Y$ in the model.",
        "Then it converges.",
    ]


def test_closing_quotes_stay_with_their_sentence():
    assert split_sentences('He wrote "it fails." Then he fixed it.') == [
        'He wrote "it fails."',
        "Then he fixed it.",
    ]


def test_text_without_a_boundary_is_one_sentence():
    assert split_sentences("## A heading") == ["## A heading"]
    assert split_sentences("") == []

"""A source's name written into a section heading is moved into the claim."""


def test_an_attribution_in_the_heading_moves_into_the_claim():
    """A survey answer headed a section "Jiang et al. (2026)" while its claim
    named nobody. The heading is a direction shared by the section's claims;
    the claim is where its own source is named."""
    from app.domain.attribution import move_attribution_out_of_theme

    theme, text = move_attribution_out_of_theme(
        "Adaptive correction — Jiang et al. (2026)",
        "A method restores the focus after the fiber bends.",
    )

    assert theme == "Adaptive correction"
    assert text == "A method restores the focus after the fiber bends (Jiang et al., 2026)."


def test_an_attribution_the_claim_already_gives_is_only_dropped_from_the_heading():
    from app.domain.attribution import move_attribution_out_of_theme

    theme, text = move_attribution_out_of_theme(
        "Jiang et al. (2026): adaptive correction",
        "Jiang et al. (2026) restore the focus after the fiber bends.",
    )

    assert theme == "adaptive correction"
    assert text == "Jiang et al. (2026) restore the focus after the fiber bends."


def test_a_heading_that_merely_mentions_a_year_is_not_an_attribution():
    from app.domain.attribution import move_attribution_out_of_theme

    for heading in ("QLoRA 2023 variants", "Benchmarks since 2020", "GPT 2024 models"):
        assert move_attribution_out_of_theme(heading, "Text.") == (heading, "Text.")

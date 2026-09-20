from app.providers.latex import restore_latex


def test_restores_backslashes_inside_inline_math():
    text = "The tensor $G_{@mu@nu}$ is symmetric."
    assert restore_latex(text) == r"The tensor $G_{\mu\nu}$ is symmetric."


def test_restores_backslashes_inside_display_math():
    text = "Written out: $$R_{@mu@nu} - @frac{1}{2} R g_{@mu@nu} = @kappa T_{@mu@nu}$$"
    assert restore_latex(text) == (
        r"Written out: $$R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} = \kappa T_{\mu\nu}$$"
    )


def test_leaves_at_sign_outside_math_alone():
    text = "Contact a@b.com about $x@alpha$."
    assert restore_latex(text) == r"Contact a@b.com about $x\alpha$."


def test_strips_control_characters_left_by_mangled_escapes():
    # \x0b is what Gemini collapses \mu\nu into in structured output.
    text = "The tensor $R_{\x0b}$ and $\tilde{x}$ survive as renderable text."
    assert restore_latex(text) == "The tensor $R_{}$ and $ilde{x}$ survive as renderable text."


def test_keeps_newlines_in_prose():
    text = "First line.\nSecond line."
    assert restore_latex(text) == "First line.\nSecond line."


def test_plain_text_is_unchanged():
    text = "No mathematics here at all."
    assert restore_latex(text) == text

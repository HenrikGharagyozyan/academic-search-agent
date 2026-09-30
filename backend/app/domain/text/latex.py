"""Getting LaTeX through a channel that eats backslashes.

The model's answer is a JSON string, and in JSON a backslash starts an escape.
A command the model writes as ``\\text`` arrives as a tab followed by "ext",
``\\frac`` as a form feed and "rac", ``\\nu`` as a newline and "u"; a command
whose first letter is no JSON escape at all comes out as whatever the decoder
could make of it, down to a lone control character. Stripping those characters
then left ``$S = k_B ext{ln} ext{Ω}$`` on the page, and ``$f$`` where an Ω had
been.

So the channel uses ``@`` as the command character, in both directions. The
passages are rewritten with it before the model sees them — it copies what it
reads, and it was reading backslashes — the model is asked to write with it,
and the backslash is put back here.

The model does not always comply, so what JSON ate is also repaired where the
damage is reversible, and reported where it is not.
"""

import re

from app.domain.text.fences import outside_fences

# Inline $...$ and block $$...$$ math expressions. For inline math, allow any
# characters except $, so we can catch fragments already corrupted by control characters.
_MATH_SPAN = re.compile(r"\$\$.+?\$\$|\$[^$]{1,400}?\$", re.DOTALL)

# C0 control characters except newline: in model text, these are always traces of
# decoded escape sequences rather than meaningful content.
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f]")

# Command names, to tell what a control character used to be. JSON allows a
# backslash to be followed only by one of its own escape letters, so a decoder
# held to valid JSON turns the start of "\Omega" into whichever escape it can:
# sometimes the escape swallows the first letter ("\text" -> tab, "ext"),
# sometimes the name survives behind it ("\x1aDelta", "\x08aOmega").
_COMMANDS = frozenset(
    """
    alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
    lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega
    Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
    text textbf textit mathrm mathbf mathit mathsf mathcal mathbb boldsymbol operatorname
    frac tfrac dfrac binom sqrt sum prod int oint partial nabla infty hbar ell
    times cdot cdots ldots dots div pm mp otimes oplus circ star ast dagger
    leq geq le ge neq ne approx sim simeq equiv propto ll gg
    ln log exp sin cos tan sinh cosh tanh lim max min sup inf det dim ker arg Pr
    left right big Big langle rangle lvert rvert vert mid parallel perp angle
    rightarrow leftarrow Rightarrow Leftarrow leftrightarrow to mapsto
    in notin subset subseteq supset supseteq cup cap forall exists neg land lor top bot
    hat bar vec tilde dot ddot overline underline boxed quad qquad
    begin end displaystyle textstyle
    """.split()
)

# The escapes that swallow the first letter of a command, and the letter each ate.
_EATEN = {"\t": "t", "\x0c": "f", "\x08": "b", "\r": "r", "\x0b": "v", "\n": "n"}

_CONTROL_THEN_LETTERS = re.compile(r"([\x00-\x1f])([a-zA-Z]+)")

# One or more backslashes in front of a command name. Scraped Markdown doubles
# them, which is the same command.
_BACKSLASH_COMMAND = re.compile(r"\\+(?=[a-zA-Z])")


def to_channel_notation(text: str) -> str:
    """Rewrites LaTeX commands with ``@`` for text that is about to be shown to
    the model, so that the notation it reads is the notation it must write."""
    return outside_fences(text, lambda prose: _BACKSLASH_COMMAND.sub("@", prose))


def restore_latex(text: str) -> str:
    """Removes stray control characters and restores backslashes inside math expressions.

    Fenced blocks are left alone: a diagram's @ is an @, and its alignment is
    not a stray control character.
    """
    return outside_fences(text, _restore_in_prose)


def has_damaged_maths(text: str) -> bool:
    """True when a formula holds a control character that cannot be traced back
    to the command it came from — the formula is wrong, not merely untidy."""
    found = False

    def look(prose: str) -> str:
        nonlocal found
        found = found or any(
            _CONTROL_CHARS.search(_repair(span.group())) for span in _MATH_SPAN.finditer(prose)
        )
        return prose

    outside_fences(text, look)
    return found


def _repair(span: str) -> str:
    """Puts back the backslash commands JSON decoded into control characters.

    Only where the result is a command that exists: a newline before "eq" was
    "\neq", a newline before anything else is a newline.
    """

    def put_back(match: re.Match) -> str:
        control, letters = match.groups()
        eaten = _EATEN.get(control, "") + letters
        if control in _EATEN and eaten in _COMMANDS:
            return "\\" + eaten
        if letters in _COMMANDS:
            return "\\" + letters
        # One stray letter between the escape and an intact name: "\x08aOmega".
        if letters[1:] in _COMMANDS:
            return "\\" + letters[1:]
        return match.group()

    return _CONTROL_THEN_LETTERS.sub(put_back, span)


def _restore_in_prose(text: str) -> str:
    repaired = _MATH_SPAN.sub(lambda m: _repair(m.group()), text)
    cleaned = _CONTROL_CHARS.sub("", repaired)
    return _MATH_SPAN.sub(lambda m: m.group(0).replace("@", "\\"), cleaned)

"""Finding equations in text, however they were typeset.

Two questions need answering about an answer: did an equation the passages
state make it into the text at all, and is it set as mathematics or left as
plain characters. Both compare the same equation across notations — "F = ma",
"$$F = ma$$", "S=k_{\\text{B}}\\ln \\Omega", "S = k_B ln Ω" — so both reduce
an expression to a skeleton that ignores how it was written.
"""

import re

from app.domain.text.fences import outside_fences

_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$([^$\n]+?)\$", re.DOTALL)

# What makes an expression an equation rather than a symbol: a relation.
_RELATION = re.compile(r"=|≥|≤|≈|∝|\\(?:geq?|leq?|approx|propto|sim|equiv)\b")

# Commands that change how something looks and not what it says.
_STYLING = re.compile(
    r"[\\@](?:text|mathrm|mathsf|mathbf|mathit|mathcal|operatorname|boldsymbol|vec|bar|hat"
    r"|left|right|displaystyle|textstyle|quad|qquad|cdot|times|,|;|!|:)(?![a-zA-Z])"
)
_COMMAND = re.compile(r"[\\@]([a-zA-Z]+)")

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "geq": "≥", "ge": "≥", "leq": "≤", "le": "≤", "approx": "≈", "propto": "∝",
    "partial": "∂", "infty": "∞",
}
_SCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉", "0123456789-+0123456789")
# Characters that carry no meaning of their own once the structure is flattened.
_NOISE = re.compile(r"[\s{}_^·×*⋅⁡⁢]")


def skeleton(expression: str) -> str:
    """An expression with its typesetting removed: ``S=kBlnΩ`` for every way of
    writing Boltzmann's formula."""
    text = _STYLING.sub("", expression)
    text = _COMMAND.sub(lambda m: _GREEK.get(m.group(1), m.group(1)), text)
    return _NOISE.sub("", text.translate(_SCRIPTS))


def math_spans(text: str) -> list[str]:
    """The contents of every ``$...$``, ``$$...$$`` and ``\\[...\\]`` in prose."""
    spans: list[str] = []

    def collect(prose: str) -> str:
        spans.extend(next(g for g in m.groups() if g is not None) for m in _MATH_SPAN.finditer(prose))
        return prose

    outside_fences(text, collect)
    return spans


def display_equations(text: str) -> list[str]:
    """Equations a passage sets on their own, the way a page sets one that matters."""
    found: list[str] = []

    def collect(prose: str) -> str:
        for match in _MATH_SPAN.finditer(prose):
            body = match.group(1) or match.group(2)
            if body and _RELATION.search(body):
                found.append(body.strip())
        return prose

    outside_fences(text, collect)
    return found


def states_an_equation(text: str) -> bool:
    """True when the text sets at least one equation as mathematics."""
    return any(_RELATION.search(span) for span in math_spans(text))


def typesets(text: str, equation: str) -> bool:
    """True when ``equation`` appears in ``text`` inside maths delimiters."""
    wanted = skeleton(equation)
    return any(wanted in skeleton(span) for span in math_spans(text))


def mentions_in_plain_text(text: str, equation: str) -> bool:
    """True when ``equation`` appears in ``text`` outside any maths delimiters."""
    return skeleton(equation) in skeleton(_MATH_SPAN.sub(" ", text))


_URL = re.compile(r"(?:https?://|www\.)\S+")

# An equation written as ordinary characters: a short symbol, a relation, and
# something to relate it to — "E = mc²", "F=ma", "S = k_B ln Ω", "ΔS ≥ 0".
# A number on the right of an equals sign is left out on purpose: "r = 8" and
# "n = 120" are settings reported in prose, not equations a page would set.
_SYMBOL = r"(?<![\w$@\\/=])(?:[A-Za-z]|Δ[A-Za-z]|[A-Za-z]_[A-Za-z0-9]{1,3})"
_PLAIN_EQUATION = re.compile(
    rf"{_SYMBOL}\s*(?:=\s*[A-Za-zΔΩ(√∂]|[≥≤≈∝]\s*[A-Za-z0-9ΔΩ(√∂])[^\s,;:)]*"
)


def plain_text_equations(text: str) -> list[str]:
    """Equations the text states outside maths delimiters, where the interface
    shows them as the characters they are instead of setting them."""
    found: list[str] = []

    def collect(prose: str) -> str:
        bare = _URL.sub(" ", _MATH_SPAN.sub(" ", prose))
        found.extend(match.group().strip() for match in _PLAIN_EQUATION.finditer(bare))
        return prose

    outside_fences(text, collect)
    return found

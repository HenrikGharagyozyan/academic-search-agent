"""Restores LaTeX in text produced by the model.

The backslash does not survive Gemini's structured output: sequences like
``\\mu\\nu`` are collapsed into a control character (``\\x0b``) before the
response even reaches us, and local parsing cannot fix that. For this reason,
the model is asked to write commands using ``@`` (``@frac``, ``@mu``), and the
backslash is restored here.
"""

import re

# Inline $...$ and block $$...$$ math expressions. For inline math, allow any
# characters except $, so we can catch fragments already corrupted by control characters.
_MATH_SPAN = re.compile(r"\$\$.+?\$\$|\$[^$]{1,400}?\$", re.DOTALL)

# C0 control characters except newline: in model text, these are always traces of
# decoded escape sequences rather than meaningful content.
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f]")


def restore_latex(text: str) -> str:
    """Removes stray control characters and restores backslashes inside math expressions."""
    cleaned = _CONTROL_CHARS.sub("", text)
    return _MATH_SPAN.sub(lambda m: m.group(0).replace("@", "\\"), cleaned)

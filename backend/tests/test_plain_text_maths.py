"""Maths is typeset whatever form the source gave it in.

"What is energy in physics? what about formula Einstein" carried the formula
as the unicode text "E = mc²". The rule that every expression is wrapped in
LaTeX was already in the prompt; the model applied it only when the source had
written LaTeX itself, and copied a popular article's plain text as it stood.
"""

from app.infrastructure.llm.prompts import ANSWER_PROMPT, SYSTEM_PROMPT


def rendered() -> str:
    """The prompt as the model receives it, template braces resolved."""
    messages = ANSWER_PROMPT.invoke({"question": "q", "evidence_block": "e"}).to_messages()
    return " ".join(messages[0].content.split())


def test_plain_text_maths_is_converted_like_any_other():
    prompt = rendered()
    assert "PLAIN-TEXT MATHS IS STILL MATHS" in prompt
    assert '"E = mc²"' in prompt
    assert "$$E = mc^2$$" in prompt
    assert "only the source's CONTENT" in prompt


def test_the_plain_text_rule_sits_with_the_rule_it_extends():
    carry = SYSTEM_PROMPT.index("CARRY THE EVIDENCE'S MATHS ACROSS")
    plain = SYSTEM_PROMPT.index("PLAIN-TEXT MATHS IS STILL MATHS")
    diagrams = SYSTEM_PROMPT.index("PROCESS DIAGRAMS")
    assert carry < plain < diagrams


def test_the_new_latex_examples_survive_the_template():
    # A single brace in the template is a variable; the example has to reach
    # the model as real LaTeX, in the @ notation the channel uses.
    assert "$@sqrt{x}$" in rendered()


def test_a_fragment_inside_a_sentence_stays_inline():
    assert 'never "the factor $$c^2$$"' in rendered()

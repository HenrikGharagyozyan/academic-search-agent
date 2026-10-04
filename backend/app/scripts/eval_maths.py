"""Live check that equations stated in plain text come back as LaTeX.

    uv run python -m app.scripts.eval_maths [runs]

Runs the generation step on fixed passages that state their equations the way
a popular article does — "F = ma", "S = k_B ln Ω", "E = mc²" — and prints the
model's raw output for every attempt next to a verdict on each equation in the
answer the reader would get: typeset, left as plain text, or missing. Exits
non-zero unless every run typesets every one.

The raw output is printed before any post-processing on purpose. When a formula
is wrong on the page there are three suspects — the model, the filters that
restore LaTeX and strip ids, and the renderer — and only the raw text says
which.
"""

import re
import sys

from app.application.agents.nodes import generate_claims_node
from app.domain.documents import Chunk
from app.domain.text.equations import mentions_in_plain_text, typesets
from app.infrastructure.llm import create_llm_provider

# Superscripts, subscripts and operators that should only ever appear as LaTeX.
_BARE_MATHS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂₃₄₅₆₇₈₉≥≤√]")
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)

CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "What is Newton's second law of motion?",
        [
            "Newton's second law of motion states that the acceleration of an object is "
            "directly proportional to the net force acting on it and inversely proportional "
            "to its mass. The law is usually written F = ma, where F is the net force in "
            "newtons, m is the mass in kilograms and a is the acceleration in m/s².",
            "Rearranged, a = F/m: for a fixed force, doubling the mass halves the "
            "acceleration. Newton himself stated the law in terms of momentum, and for a "
            "constant mass that statement reduces to F = ma.",
        ],
        ["F = ma"],
    ),
    (
        "What is entropy in statistical mechanics?",
        [
            "In statistical mechanics, Boltzmann's entropy formula relates the entropy S of "
            "a macrostate to the number of microstates Ω consistent with it: S = k_B ln Ω, "
            "where k_B is the Boltzmann constant. The more microstates, the higher the entropy.",
            "In classical thermodynamics Clausius defined entropy through heat and "
            "temperature. The second law states that for an isolated system ΔS ≥ 0: entropy "
            "never decreases.",
        ],
        ["S = k_B ln Ω"],
    ),
    (
        "What is energy in physics, and what is Einstein's formula?",
        [
            "Energy is the capacity of a system to do work, measured in joules.",
            "Einstein's mass–energy equivalence, E = mc², says that mass and energy are the "
            "same thing in different units: c is the speed of light.",
        ],
        ["E = mc²"],
    ),
]


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"ev_{index}", document_id="eval", text=text,
        start_line=1, end_line=1, source_url="https://example.com", title="Eval",
    )


class _Recording:
    """The provider, keeping what the model returned before anything touched it."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self.raw = []

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def generate_answer(self, question, evidence):
        result = self._llm.generate_answer(question, evidence)
        self.raw.append(result)
        return result


def main(runs: int) -> int:
    llm = _Recording(create_llm_provider())
    print(f"provider={llm.provider_name} model={llm.model_name} runs={runs}\n")
    failures = 0

    for question, passages, equations in CASES:
        evidence = [_chunk(i, text) for i, text in enumerate(passages, start=1)]
        for run in range(1, runs + 1):
            llm.raw.clear()
            out = generate_claims_node({"question": question, "selected_chunks": evidence}, llm=llm)
            final = "\n".join([out["summary"], *(c.text for c in out["claims"]), out["conclusion"]])

            print(f"=== {question}  (run {run}/{runs}, {len(llm.raw)} generation(s))")
            for number, raw in enumerate(llm.raw, start=1):
                print(f"RAW[{number}] summary:   ", repr(raw.summary))
                for claim in raw.claims:
                    print(f"RAW[{number}] claim[{claim.theme}]:", repr(claim.text))
                print(f"RAW[{number}] conclusion:", repr(raw.conclusion))

            for equation in equations:
                if typesets(final, equation):
                    verdict = "typeset"
                elif mentions_in_plain_text(final, equation):
                    verdict = "PLAIN TEXT"
                else:
                    verdict = "MISSING"
                failures += verdict != "typeset"
                print(f"  {equation!r}: {verdict}")

            bare = sorted(set(_BARE_MATHS.findall(_MATH.sub(" ", final))))
            if bare:
                print(f"  bare maths characters outside LaTeX: {' '.join(bare)}")
            print()

    print("FAILED" if failures else "OK", f"— {failures} equation(s) not typeset")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))

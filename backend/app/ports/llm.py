"""The language model as the research pipeline needs it.

Every node depends on this Protocol, never on a vendor class. Two concrete
providers already drifted apart once — one read ``AIMessage.content``, the other
``AIMessage.text``, and only one of them survived a model that returns content
blocks. A single declared surface is what keeps that from recurring.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.answers import Claim, ClaimsResponse
from app.domain.documents import Chunk
from app.domain.grading import AnswerQualityGrade, RelevanceGrade


@runtime_checkable
class LLMProvider(Protocol):
    """Generation and grading. Embeddings are a separate port."""

    @property
    def provider_name(self) -> str:
        """Registry key this provider was built from, for logs and diagnostics."""

    @property
    def model_name(self) -> str:
        """The model actually in use, as the vendor names it."""

    def generate_answer(self, question: str, evidence: Sequence[Chunk]) -> ClaimsResponse:
        """Writes a grounded answer citing only the given chunks."""

    def refine_query(self, question: str, previous_query: str) -> str:
        """Rewrites a search query that did not yield usable evidence."""

    def grade_relevance(self, question: str, chunks: Sequence[Chunk]) -> RelevanceGrade:
        """Picks the chunks that genuinely bear on the question."""

    def grade_answer_quality(
        self, question: str, summary: str, claims: Sequence[Claim], conclusion: str
    ) -> AnswerQualityGrade:
        """Judges whether the answer is a satisfactory response to the question."""

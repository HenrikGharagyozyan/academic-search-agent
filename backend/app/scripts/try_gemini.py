"""Manual smoke check against the configured live LLM provider."""

from app.domain.documents import Chunk
from app.infrastructure.llm import create_llm_provider


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="manual",
        text=text,
        start_line=1,
        end_line=1,
        source_url="https://example.com",
        title="Manual check",
    )


if __name__ == "__main__":
    llm = create_llm_provider()
    print(f"provider={llm.provider_name} model={llm.model_name}")

    result = llm.generate_answer(
        "What is gradient descent?",
        [
            _chunk("ev_1", "Gradient descent iteratively steps against the gradient."),
            _chunk("ev_2", "For convex objectives it converges to the global minimum."),
        ],
    )
    print(result.model_dump_json(indent=2))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import get_settings


class EmbeddingProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
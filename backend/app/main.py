import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.search import router as search_router
from app.api.routes.documents import router as documents_router
from app.api.routes.answer import router as answer_router
from app.infrastructure.search.firecrawl import FirecrawlProvider
from app.application.services.document import DocumentService
from app.application.services.research import ResearchService
from app.application.services.search import SearchService
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    firecrawl = FirecrawlProvider()
    app.state.research_service = ResearchService(firecrawl=firecrawl)
    app.state.search_service = SearchService(provider=firecrawl)
    app.state.document_service = DocumentService(provider=firecrawl)
    yield


app = FastAPI(title="Academic Search Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(documents_router)
app.include_router(answer_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
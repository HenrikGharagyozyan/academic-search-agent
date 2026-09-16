from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.search import router as search_router
from app.api.routes.documents import router as documents_router
from app.api.routes.answer import router as answer_router

app = FastAPI(title="Academic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(documents_router)
app.include_router(answer_router)


@app.get("/health")
async def health():
    return {"status": "ok"}

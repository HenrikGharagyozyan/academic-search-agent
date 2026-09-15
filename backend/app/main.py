from fastapi import FastAPI

from app.api.routes.search import router as search_router
from app.api.routes.documents import router as documents_router

app = FastAPI(title="Academic Search")

app.include_router(search_router)
app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
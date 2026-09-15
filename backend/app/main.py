from fastapi import FastAPI

from app.api.routes.search import router as search_router

app = FastAPI(title="Academic Search")

app.include_router(search_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
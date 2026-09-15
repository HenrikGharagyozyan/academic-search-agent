from fastapi import FastAPI

app = FastAPI(title="AgentX Search")

@app.get("/health")
async def health():
    return {"status": "ok"}
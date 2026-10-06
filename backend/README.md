# Backend

FastAPI + LangGraph research pipeline for Academic Search Agent.

```bash
cp .env.example .env        # fill in FIRECRAWL_API_KEY and GEMINI_API_KEY
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
```

Setup, configuration, the pipeline and the layer rules are documented in the [root README](../README.md).

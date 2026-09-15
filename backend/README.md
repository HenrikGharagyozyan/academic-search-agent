# Academic Search — Academic Research Agent

AI-powered research assistant that finds academic sources, retrieves their
content, and answers research questions with **grounded, verifiable
citations** — every claim in the answer links back to the exact passage it
came from.

## Why

Regular LLM chat answers can't be trusted for research: they sound
confident but can hallucinate sources or facts. AgentX Search fixes this by
separating **evidence retrieval** from **answer generation**:

1. Search the web / academic sources for relevant pages.
2. Scrape the full content (not just a snippet).
3. Extract evidence passages with exact source metadata (URL, title, line
   range).
4. Ask the LLM to answer *using only that evidence*, citing which passage
   supports each claim.
5. Render the answer with clickable/hoverable citation markers that show
   the exact quoted text and a link to the source.

## Tech stack

| Layer            | Technology              |
|-------------------|--------------------------|
| Language          | Python 3.12+             |
| Package manager   | uv                        |
| Backend           | FastAPI                  |
| Validation        | Pydantic                 |
| Search / Scrape   | Firecrawl                |
| LLM               | Gemini API                |
| Frontend          | React + TypeScript        |
| Tests             | pytest                   |

## Project status

Early stage — building the backend foundation first, then the
search/evidence pipeline, then the agent orchestration, then the UI.
See `docs/` for architecture notes as they're added.

## Getting started (backend)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) installed.

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Check it's alive:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

### Running tests

```bash
cd backend
uv run pytest
```

## Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable          | Description                        |
|--------------------|-------------------------------------|
| `FIRECRAWL_API_KEY` | API key for Firecrawl search/scrape |
| `GEMINI_API_KEY`    | API key for Google Gemini           |

## Project structure

```
agentx-search/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app entrypoint
│   │   ├── api/routes/      # HTTP route handlers
│   │   ├── core/            # settings, config
│   │   ├── schemas/         # Pydantic models
│   │   ├── services/        # business logic
│   │   ├── agents/          # LangGraph agent (later)
│   │   ├── providers/       # Firecrawl / Gemini clients
│   │   ├── retrieval/       # document parsing, chunking
│   │   └── citations/       # evidence & citation logic
│   └── tests/
├── frontend/                 # React + TypeScript UI
├── docs/                      # architecture & design notes
└── .github/workflows/        # CI
```

## Contributing / workflow

This project follows a simple Git flow:

- `main` — stable, released state only
- `develop` — integration branch for finished features
- `feat/<name>` — one feature per branch, opened as a PR into `develop`

No direct commits to `main` or `develop`.
# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-09-24

A stabilisation release: no new user-facing features, but the retrieval path
was producing far worse evidence than intended, and the backend had no declared
structure to hang a second LLM vendor on.

### Fixed

- **Chunking produced one chunk per markdown paragraph.** Firecrawl emits one
  paragraph or heading per line separated by blank lines, so the line-count
  limit never triggered: a 40-line page became 40 chunks, bare headings among
  them. Chunks are now packed to a character budget with overlap, and a heading
  ships with the section it introduces. The same page now yields 15 chunks of
  ~1100 characters with no orphan headings.
- **A paragraph larger than the budget was emitted whole.** A 60k-character
  scraped line became a single chunk 43× over budget, overrunning the embedding
  model's input limit — and a failed embedding call drops the whole request back
  to "first N chunks", disabling semantic retrieval for every source. Oversized
  paragraphs are split on word boundaries.
- **Chunks were emitted twice.** The overlap carried a whole paragraph forward
  even when the buffer held only that one paragraph, measured at 1.79× text
  amplification. Duplicates cost embedding calls and competed for the top-k
  slots.
- **Evidence-id stripping ran words together.** `"Adam <id> converges"` became
  `"Adamconverges"` in the displayed answer.
- **A silent stream could leave the UI blank.** A failure while assembling the
  answer escaped after the response headers were sent, so no error event
  reached the client.
- **`OPENROUTER` answers could not be refined.** The OpenRouter adapter read
  `AIMessage.content`, which is a list for models returning content blocks;
  refinement failed and the retry loop silently repeated the first query.
- **A misspelled `LLM_PROVIDER` fell back to Gemini** instead of failing, and
  selecting OpenRouter without a key failed only on the first request.
- **The embedding cache was unbounded** — roughly 96 KiB per vector, for the
  life of the process — and keyed on text alone, so a passage and a question
  with identical text shared one entry despite being embedded under different
  task types.
- No timeouts on any outbound call.

### Changed

- **The backend is laid out in layers** — `domain`, `ports`, `infrastructure`,
  `application`, `api` — with the dependency direction enforced by a test that
  reads each module's imports.
- **LLM providers sit behind one `LLMProvider` port.** Both vendor adapters
  derive from a shared base holding prompt assembly, structured output and the
  retry policy, and are built through a registry. Adding a vendor is an adapter
  plus one registry entry.
- Retry policy is unified: the two providers previously carried disagreeing
  lists of retryable error markers.
- The model, the embedding model and the CORS origins are configuration rather
  than literals in the source.
- CI now builds and lints the frontend; previously only backend tests ran.

### Removed

- `askQuestion` on the frontend, unused since streaming landed. The `POST
  /api/v1/answer` endpoint it called remains as documented API.

### Configuration

`OPENROUTER_MODEL` is replaced by `LLM_MODEL`, which applies to whichever
provider is selected. Leaving it empty uses that provider's default. See
`backend/.env.example`.

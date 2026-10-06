# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-10-04

Breadth and trust. An answer now draws on several differently-aimed searches,
reads the pages most worth reading rather than the first twelve, cites them by
name and year as the sources themselves record it, and shows the reader how it
was found. The retry loop knows why the first answer fell short and never
loses it.

### Added

- **How an answer was found.** Every step — the searches planned, each page
  found, read or skipped, how passages were ranked and judged, what was
  verified — is streamed while the run is in progress and kept on the answer.
- **Several searches per question.** The model plans up to four queries that
  approach the question from different directions; one that leaves the
  question's subject is discarded rather than searched.
- **Choosing which pages to read.** Each query returns ten results, and twelve
  are picked from all of them before anything is scraped: scholarly hosts first,
  then reference and documentation, with Reddit, YouTube, Medium and the like
  demoted but never excluded; relevance to the question and the engine's own
  ranking count for more than the host. Copies of one paper on several sites
  are set aside.
- **Sources' own authors and year.** A page's citation tags are read when it is
  scraped, each passage is shown to the model with them, and an attribution's
  year is set from them.
- **Citations as site chips.** A claim's sources appear after its last sentence
  as chips with the site's icon and name, one per page, each opening the
  passages it cites.
- **Dark theme**, with a system / light / dark switch that is remembered.
- `app.scripts.eval_sources`, which captures searches and compares which pages
  two versions of the code would read, and runs control questions end to end.

### Changed

- **Answers take the shape of their evidence.** A direct question gets one
  prose answer; a survey question gets claims grouped under numbered themes,
  with a conclusion that synthesises across them.
- **Context is spread across sources.** Each page gets a turn before any page
  repeats, so one long page cannot fill the context; `TOP_K_CHUNKS` is 60.
- **A retry aims at what was missing.** The answer grader says why an answer
  fell short — off topic, too thin, missing an aspect — and what it needed, and
  the rewritten search is given that and the searches already run. A question
  the grader judges a literature search cannot answer is not searched again;
  when no claim survives verification the grader is not asked, and the retry
  still runs.

### Fixed

- **The same source was cited with different years** in two runs of one
  question ("Cheng et al. (2022)" and "(2023)"). The year now comes from the
  source's own record.
- **A source named in a section heading was missing from the claim** beneath
  it. The claim is now the one place a source is named.
- **A retry could take the answer away.** When the second pass found nothing,
  or every search failed, the run ended empty or with an error in place of the
  first pass's grounded answer. That answer is now kept.
- **The attribution rule's example names were cited as sources**, in answers on
  unrelated topics. The examples are placeholders.
- **"Newton" read as a question about recent work**, because "new" matched as a
  prefix, and the search was limited to the last three years.
- **A search refused for quota was not retried**, though scrapes were; a
  planned query could silently drop out.
- **Formulas.** JSON no longer eats the model's LaTeX, an equation written as
  plain text is typeset, Wikipedia's formulas arrive once and as TeX, and an
  answer that drops every equation the passages state is regenerated.
- **Without embeddings, passages were taken in page order.** They are ranked
  by the question's words instead.
- **An out-of-credits model provider** surfaced as a generic failure; the
  reader is now told.

### Known limitations

- The planner writes queries in the vocabulary its model knows, and may miss
  terms a fast-moving field adopted since.
- A claim can name an author whose page it does not itself cite; only the year
  of an author it does cite is checked.

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

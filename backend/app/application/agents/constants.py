NODE_PLAN_SEARCHES = "plan_searches"
NODE_SEARCH = "search"
NODE_RETRIEVE_AND_CHUNK = "retrieve_and_chunk"
NODE_GENERATE_CLAIMS = "generate_claims"
NODE_VERIFY_EVIDENCE = "verify_evidence"
NODE_REFINE_QUERY = "refine_query"
NODE_SELECT_CHUNKS = "select_relevant_chunks"
NODE_GRADE_RELEVANCE = "grade_relevance"
NODE_GRADE_ANSWER = "grade_answer"

STAGE_LABELS: dict[str, str] = {
    NODE_PLAN_SEARCHES: "Planning the search",
    NODE_SEARCH: "Searching sources",
    NODE_RETRIEVE_AND_CHUNK: "Reading sources",
    NODE_SELECT_CHUNKS: "Selecting relevant passages",
    NODE_GRADE_RELEVANCE: "Filtering relevant passages",
    NODE_GENERATE_CLAIMS: "Generating answer",
    NODE_VERIFY_EVIDENCE: "Verifying claims against sources",
    NODE_GRADE_ANSWER: "Reviewing answer quality",
    NODE_REFINE_QUERY: "Refining search query",
}

ROUTE_END = "end"
ROUTE_REFINE = "refine"

# How many differently-aimed searches to run per attempt, and how many results
# to take from each. The budget is deliberately spread: one query returning six
# results is how a topic came back as one paper under six domains. Ten per query
# rather than five so that there is something to choose between: the twelve
# pages read are picked from all of them by app.domain.sources, and the journal
# article at position seven is no longer out of reach behind a forum thread.
MAX_QUERIES = 4
RESULTS_PER_QUERY = 10

# Distinct pages to read per attempt. The mirror check collapses duplicates of
# one paper, so a page here is closer to a distinct document than it used to be,
# and the relevance judge discards most of what arrives — both argue for asking
# for more than the eight this started at. Unchanged by the wider search above:
# this is bounded by the provider's scrape quota, not by how many candidates
# there are.
MAX_SOURCES = 12
MAX_RETRIES = 1

# How many passages the relevance judge sees per call. Measured on a fixed set of
# forty passages: one call kept 7-8 of them, batches of ten kept 15-17, batches
# of eight kept 19-20. The judge cannot attend to forty heterogeneous passages at
# once, and no wording of the prompt changed that — rewriting it moved the count
# by one. Smaller batches cost more requests but the same tokens.
GRADE_BATCH_SIZE = 8

# Passages handed to the relevance judge and then to the model. Raised again
# because the judge is where the funnel narrows hardest — it discards most of
# what it sees — so the number of candidates it gets is what decides how many
# approaches the answer can cover. Embedding cost does not scale with this:
# every passage is embedded either way, so the price is generation tokens.
#
# Raised from 40 when the pages read became distinct and readable. Passages are
# shared out a turn per source, so the budget is divided by the number of pages
# that yield any: before source ranking, half the queue was Reddit and YouTube,
# which Firecrawl cannot read or which come back empty, and about six pages split
# forty passages. With twelve real pages each got three, and Wikipedia's one
# passage on "what is entropy" carried no formula — measured on two runs, the
# formulas in the answer fell from fourteen to six. Sixty gives each page about
# five again. The batches stay at GRADE_BATCH_SIZE and run in parallel, so the
# judge is asked more often, not asked to weigh more at once.
TOP_K_CHUNKS = 60

# Below the provider's per-minute scrape budget on purpose: twelve at once
# exhausted it and the refused pages were reported as unreadable. The retry in
# the adapter covers an occasional overshoot; this keeps the burst from causing
# one in the first place.
MAX_SCRAPE_WORKERS = 6
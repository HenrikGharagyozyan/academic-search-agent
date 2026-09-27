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
# results is how a topic came back as one paper under six domains.
MAX_QUERIES = 4
RESULTS_PER_QUERY = 4

# Distinct pages to read per attempt. Raised from 6 with the move to several
# queries, so that breadth is not bought by dropping the canonical sources.
MAX_SOURCES = 8
MAX_RETRIES = 1
TOP_K_CHUNKS = 25

# For a recency question, how far back still counts as current work. A year is
# the usual web default and too tight for a literature review.
RECENCY_WINDOW_YEARS = 3

MAX_SCRAPE_WORKERS = 8
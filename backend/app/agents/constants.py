NODE_SEARCH = "search"
NODE_RETRIEVE_AND_CHUNK = "retrieve_and_chunk"
NODE_GENERATE_CLAIMS = "generate_claims"
NODE_VERIFY_EVIDENCE = "verify_evidence"
NODE_REFINE_QUERY = "refine_query"
NODE_SELECT_CHUNKS = "select_relevant_chunks"
NODE_GRADE_RELEVANCE = "grade_relevance"
NODE_GRADE_ANSWER = "grade_answer"

STAGE_LABELS: dict[str, str] = {
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

MAX_SOURCES = 6
MAX_RETRIES = 1
TOP_K_CHUNKS = 25

MAX_SCRAPE_WORKERS = 6
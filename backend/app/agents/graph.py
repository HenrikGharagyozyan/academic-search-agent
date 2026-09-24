from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agents.constants import (
    NODE_GENERATE_CLAIMS,
    NODE_GRADE_ANSWER,
    NODE_GRADE_RELEVANCE,
    NODE_REFINE_QUERY,
    NODE_RETRIEVE_AND_CHUNK,
    NODE_SEARCH,
    NODE_SELECT_CHUNKS,
    NODE_VERIFY_EVIDENCE,
    ROUTE_END,
    ROUTE_REFINE,
)
from app.agents.nodes import (
    generate_claims_node,
    grade_answer_node,
    grade_relevance_node,
    refine_query_node,
    retrieve_and_chunk_node,
    search_node,
    select_relevant_chunks_node,
    should_refine,
    verify_evidence_node,
)
from app.agents.state import ResearchState
from app.infrastructure.search.firecrawl import FirecrawlProvider
from app.infrastructure.llm import create_llm_provider
from app.infrastructure.llm.gemini import GeminiProvider
from app.infrastructure.vector_store.chroma import ChromaVectorStore


def build_research_graph(
    firecrawl: FirecrawlProvider | None = None,
    gemini: GeminiProvider | None = None,
    vector_store: ChromaVectorStore | None = None,
):
    firecrawl = firecrawl or FirecrawlProvider()
    gemini = gemini or create_llm_provider()
    vector_store = vector_store or ChromaVectorStore()

    graph = StateGraph(ResearchState)

    graph.add_node(NODE_SEARCH, partial(search_node, firecrawl=firecrawl))
    graph.add_node(NODE_RETRIEVE_AND_CHUNK, partial(retrieve_and_chunk_node, firecrawl=firecrawl))
    graph.add_node(NODE_SELECT_CHUNKS, partial(select_relevant_chunks_node, vector_store=vector_store))
    graph.add_node(NODE_GRADE_RELEVANCE, partial(grade_relevance_node, gemini=gemini))
    graph.add_node(NODE_GENERATE_CLAIMS, partial(generate_claims_node, gemini=gemini))
    graph.add_node(NODE_VERIFY_EVIDENCE, verify_evidence_node)
    graph.add_node(NODE_GRADE_ANSWER, partial(grade_answer_node, gemini=gemini))
    graph.add_node(NODE_REFINE_QUERY, partial(refine_query_node, gemini=gemini))

    graph.add_edge(START, NODE_SEARCH)
    graph.add_edge(NODE_SEARCH, NODE_RETRIEVE_AND_CHUNK)
    graph.add_edge(NODE_RETRIEVE_AND_CHUNK, NODE_SELECT_CHUNKS)
    graph.add_edge(NODE_SELECT_CHUNKS, NODE_GRADE_RELEVANCE)
    graph.add_edge(NODE_GRADE_RELEVANCE, NODE_GENERATE_CLAIMS)
    graph.add_edge(NODE_GENERATE_CLAIMS, NODE_VERIFY_EVIDENCE)
    graph.add_edge(NODE_VERIFY_EVIDENCE, NODE_GRADE_ANSWER)

    graph.add_conditional_edges(
        NODE_GRADE_ANSWER,
        should_refine,
        {ROUTE_END: END, ROUTE_REFINE: NODE_REFINE_QUERY},
    )
    graph.add_edge(NODE_REFINE_QUERY, NODE_SEARCH)

    return graph.compile()
from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agents.constants import (
    NODE_GENERATE_CLAIMS,
    NODE_REFINE_QUERY,
    NODE_RETRIEVE_AND_CHUNK,
    NODE_SEARCH,
    NODE_VERIFY_EVIDENCE,
    ROUTE_END,
    ROUTE_REFINE,
)
from app.agents.nodes import (
    generate_claims_node,
    refine_query_node,
    retrieve_and_chunk_node,
    search_node,
    should_refine,
    verify_evidence_node,
)
from app.agents.state import ResearchState
from app.providers.firecrawl_provider import FirecrawlProvider
from app.providers.gemini_provider import GeminiProvider


def build_research_graph(
    firecrawl: FirecrawlProvider | None = None,
    gemini: GeminiProvider | None = None,
):
    firecrawl = firecrawl or FirecrawlProvider()
    gemini = gemini or GeminiProvider()

    graph = StateGraph(ResearchState)

    graph.add_node(NODE_SEARCH, partial(search_node, firecrawl=firecrawl))
    graph.add_node(NODE_RETRIEVE_AND_CHUNK, partial(retrieve_and_chunk_node, firecrawl=firecrawl))
    graph.add_node(NODE_GENERATE_CLAIMS, partial(generate_claims_node, gemini=gemini))
    graph.add_node(NODE_VERIFY_EVIDENCE, verify_evidence_node)
    graph.add_node(NODE_REFINE_QUERY, partial(refine_query_node, gemini=gemini))

    graph.add_edge(START, NODE_SEARCH)
    graph.add_edge(NODE_SEARCH, NODE_RETRIEVE_AND_CHUNK)
    graph.add_edge(NODE_RETRIEVE_AND_CHUNK, NODE_GENERATE_CLAIMS)
    graph.add_edge(NODE_GENERATE_CLAIMS, NODE_VERIFY_EVIDENCE)

    graph.add_conditional_edges(
        NODE_VERIFY_EVIDENCE,
        should_refine,
        {ROUTE_END: END, ROUTE_REFINE: NODE_REFINE_QUERY},
    )
    graph.add_edge(NODE_REFINE_QUERY, NODE_SEARCH)

    return graph.compile()
from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import generate_claims_node, retrieve_and_chunk_node, search_node
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

    graph.add_node("search", partial(search_node, firecrawl=firecrawl))
    graph.add_node("retrieve_and_chunk", partial(retrieve_and_chunk_node, firecrawl=firecrawl))
    graph.add_node("generate_claims", partial(generate_claims_node, gemini=gemini))

    graph.add_edge(START, "search")
    graph.add_edge("search", "retrieve_and_chunk")
    graph.add_edge("retrieve_and_chunk", "generate_claims")
    graph.add_edge("generate_claims", END)

    return graph.compile()
"""
graph.py — LangGraph pipeline for the agentic retrieval branch.

After refactor: normalization is handled by the middleware.
The graph starts at intent_extraction with query_bundle already in state.

Migrated from: retrieval/agentic_retrieval/graph.py
"""

from __future__ import annotations
from langgraph.graph import END, StateGraph

from .state import AgentState
from .nodes.intent_extraction import query_intent_extraction_node_factory
from .nodes.routing import modality_routing_node
from .nodes.retrieval import parallel_retrieval_node_factory
from .nodes.fusion import candidate_fusion_node
from .nodes.rerank import frame_reranking_node
from .llm_service import LLMService
from .qdrant_search import QdrantSearchService


def build_agentic_retrieval_graph(
    llm: LLMService,
    search_service: QdrantSearchService,
    top_k_per_source: int = 20,
):
    """
    Build the agentic retrieval LangGraph.

    Input state must contain:
      {"query_bundle": QueryBundle}

    Output state contains:
      {"agent_topk": List[Candidate], ...}
    """
    builder = StateGraph(AgentState)

    # NOTE: query_normalization node is REMOVED — handled by middleware
    builder.add_node("query_intent_extraction", query_intent_extraction_node_factory(llm))
    builder.add_node("modality_routing", modality_routing_node)
    builder.add_node("parallel_retrieval", parallel_retrieval_node_factory(search_service, top_k_per_source=top_k_per_source))
    builder.add_node("candidate_fusion", candidate_fusion_node)
    builder.add_node("frame_reranking", frame_reranking_node)

    builder.set_entry_point("query_intent_extraction")
    builder.add_edge("query_intent_extraction", "modality_routing")
    builder.add_edge("modality_routing", "parallel_retrieval")
    builder.add_edge("parallel_retrieval", "candidate_fusion")
    builder.add_edge("candidate_fusion", "frame_reranking")
    builder.add_edge("frame_reranking", END)

    return builder.compile()

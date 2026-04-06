from __future__ import annotations
from langgraph.graph import END, StateGraph

from state import AgentState
from nodes.normalization import query_normalization_node
from nodes.intent_extraction import query_intent_extraction_node_factory
from nodes.routing import modality_routing_node
from nodes.retrieval import parallel_retrieval_node_factory
from nodes.fusion import candidate_fusion_node
from nodes.rerank import frame_reranking_node
from services.llm_service import LLMService
from services.qdrant_search import QdrantSearchService


def build_agentic_retrieval_graph(
    llm: LLMService,
    search_service: QdrantSearchService,
):
    builder = StateGraph(AgentState)

    builder.add_node("query_normalization", query_normalization_node)
    builder.add_node("query_intent_extraction", query_intent_extraction_node_factory(llm))
    builder.add_node("modality_routing", modality_routing_node)
    builder.add_node("parallel_retrieval", parallel_retrieval_node_factory(search_service, top_k_per_source=20))
    builder.add_node("candidate_fusion", candidate_fusion_node)
    builder.add_node("frame_reranking", frame_reranking_node)

    builder.set_entry_point("query_normalization")
    builder.add_edge("query_normalization", "query_intent_extraction")
    builder.add_edge("query_intent_extraction", "modality_routing")
    builder.add_edge("modality_routing", "parallel_retrieval")
    builder.add_edge("parallel_retrieval", "candidate_fusion")
    builder.add_edge("candidate_fusion", "frame_reranking")
    builder.add_edge("frame_reranking", END)

    return builder.compile()
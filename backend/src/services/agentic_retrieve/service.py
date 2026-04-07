"""
service.py — Entry point wrapper for the agentic retrieval pipeline.

Provides AgenticRetrieveService which initializes the LangGraph pipeline
once (singleton) and exposes a retrieve() method for the controller.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .graph import build_agentic_retrieval_graph
from .llm_service import LLMService
from .qdrant_search import QdrantSearchService

logger = logging.getLogger(__name__)


class AgenticRetrieveService:
    """
    Wraps the LangGraph agentic pipeline for use by the search controller.

    Usage:
        service = AgenticRetrieveService()
        candidates = service.retrieve(query_bundle, top_k=10)
    """

    def __init__(
        self,
        llm: Optional[LLMService] = None,
        search_service: Optional[QdrantSearchService] = None,
        top_k_per_source: int = 20,
    ):
        self.llm = llm or LLMService()
        self.search_service = search_service or QdrantSearchService()
        self.graph = build_agentic_retrieval_graph(
            self.llm,
            self.search_service,
            top_k_per_source=top_k_per_source,
        )
        logger.info(
            "AgenticRetrieveService initialized (model=%s, collection=%s)",
            self.llm.model_name,
            self.search_service.collection_name,
        )

    def retrieve(self, query_bundle: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Run the full agentic pipeline and return ranked candidates.

        Parameters
        ----------
        query_bundle : dict
            Pre-processed query bundle from the middleware containing
            raw, cleaned, lang, translated_en, rewrites.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[dict]
            List of Candidate dicts with agent_score, evidence, etc.
        """
        initial_state: Dict[str, Any] = {
            "query_bundle": query_bundle,
        }

        try:
            final_state: Dict[str, Any] = self.graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Agentic pipeline failed: %s", exc)
            return []

        if final_state.get("error"):
            logger.warning("Agentic pipeline error: %s", final_state["error"])

        candidates = final_state.get("agent_topk", [])
        return candidates[:top_k]

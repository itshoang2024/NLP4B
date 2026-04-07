"""
search_controller.py — Orchestrates agentic + heuristic retrieval,
cross-source rerank, and response building.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.schemas import SearchResponse
from src.services.agentic_retrieve.service import AgenticRetrieveService
from src.services.heuristic_retrieve.service import HeuristicRetrieveService
from src.controllers.rerank import cross_source_rerank
from src.controllers.response_builder import build_response

logger = logging.getLogger(__name__)


# ── Singleton service instances ───────────────────────────────────────────────
# Initialized lazily on first use, then reused for all requests.

_agentic_service: AgenticRetrieveService | None = None
_heuristic_service: HeuristicRetrieveService | None = None


def _get_agentic_service() -> AgenticRetrieveService:
    global _agentic_service
    if _agentic_service is None:
        logger.info("Initializing AgenticRetrieveService (singleton)...")
        _agentic_service = AgenticRetrieveService()
    return _agentic_service


def _get_heuristic_service() -> HeuristicRetrieveService:
    global _heuristic_service
    if _heuristic_service is None:
        logger.info("Initializing HeuristicRetrieveService (singleton)...")
        _heuristic_service = HeuristicRetrieveService()
    return _heuristic_service


def execute_search(query_bundle: Dict[str, Any], top_k: int = 10) -> SearchResponse:
    """
    Main search orchestration:
    1. Call agentic_retrieve for intent-aware multimodal search
    2. Call heuristic_retrieve for dense hybrid RRF search
    3. Cross-source RRF rerank to merge both branches
    4. Build final response with azure URLs

    Parameters
    ----------
    query_bundle : dict
        Pre-processed by middleware: raw, cleaned, lang, translated_en, rewrites.
    top_k : int
        Number of final results to return.

    Returns
    -------
    SearchResponse
        Formatted response ready for the frontend.
    """
    t_start = time.perf_counter()

    # ── 1. Agentic retrieval ──────────────────────────────────────────────
    t_agentic = time.perf_counter()
    try:
        agentic_service = _get_agentic_service()
        agentic_results = agentic_service.retrieve(query_bundle, top_k=top_k * 2)
    except Exception as exc:
        logger.error("Agentic retrieval failed: %s", exc)
        agentic_results = []
    agentic_ms = (time.perf_counter() - t_agentic) * 1000

    # ── 2. Heuristic retrieval ────────────────────────────────────────────
    t_heuristic = time.perf_counter()
    try:
        heuristic_service = _get_heuristic_service()
        heuristic_results = heuristic_service.retrieve(query_bundle, top_k=top_k * 2)
    except Exception as exc:
        logger.error("Heuristic retrieval failed: %s", exc)
        heuristic_results = []
    heuristic_ms = (time.perf_counter() - t_heuristic) * 1000

    # ── 3. Cross-source RRF rerank ────────────────────────────────────────
    t_rerank = time.perf_counter()
    final_candidates = cross_source_rerank(
        agentic_results=agentic_results,
        heuristic_results=heuristic_results,
        top_k=top_k,
    )
    rerank_ms = (time.perf_counter() - t_rerank) * 1000

    # ── 4. Build response ─────────────────────────────────────────────────
    total_ms = (time.perf_counter() - t_start) * 1000

    raw_query = query_bundle.get("raw", query_bundle.get("cleaned", ""))

    response = build_response(
        query=raw_query,
        candidates=final_candidates,
        latency_ms={
            "agentic_ms": round(agentic_ms, 2),
            "heuristic_ms": round(heuristic_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    )

    logger.info(
        "Search completed — agentic=%d, heuristic=%d, final=%d, %.0fms",
        len(agentic_results), len(heuristic_results),
        len(final_candidates), total_ms,
    )

    return response

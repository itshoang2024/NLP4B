"""
search_controller.py — Orchestrates agentic + heuristic retrieval,
cross-source rerank, and response building.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from src.schemas import SearchResponse
from src.services.agentic_retrieve.service import AgenticRetrieveService
from src.services.heuristic_retrieve.service import HeuristicRetrieveService
from src.controllers.rerank import cross_source_rerank
from src.controllers.response_builder import build_response

logger = logging.getLogger(__name__)


# ── Singleton service instances ───────────────────────────────────────────────
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


def _run_agentic(query_bundle: Dict[str, Any], top_k: int) -> Tuple[List, float]:
    """Run agentic branch; return (results, elapsed_ms)."""
    t = time.perf_counter()
    try:
        results = _get_agentic_service().retrieve(query_bundle, top_k=top_k)
    except Exception as exc:
        logger.error("Agentic retrieval failed: %s", exc)
        results = []
    return results, (time.perf_counter() - t) * 1000


def _run_heuristic(query_bundle: Dict[str, Any], top_k: int) -> Tuple[List, float]:
    """Run heuristic branch; return (results, elapsed_ms)."""
    t = time.perf_counter()
    try:
        results = _get_heuristic_service().retrieve(query_bundle, top_k=top_k)
    except Exception as exc:
        logger.error("Heuristic retrieval failed: %s", exc)
        results = []
    return results, (time.perf_counter() - t) * 1000


def execute_search(query_bundle: Dict[str, Any], top_k: int = 10) -> SearchResponse:
    """
    Parallel search orchestration:
      1a. agentic_retrieve  ─┐ (concurrent — wall time = max of two)
      1b. heuristic_retrieve ┘
      2.  cross_source_rerank  (RRF k=60, produces "fused" candidates)
      3.  build_response

    Latency breakdown in response:
      agentic_ms   — time spent inside agentic branch
      heuristic_ms — time spent inside heuristic branch
      rerank_ms    — cross-source RRF merge
      total_ms     — wall clock from start to response built
                     (≈ max(agentic_ms, heuristic_ms) + rerank_ms)
    """
    t_start = time.perf_counter()

    # ── 1. Run both branches in parallel ─────────────────────────────────────
    agentic_results: List = []
    heuristic_results: List = []
    agentic_ms = 0.0
    heuristic_ms = 0.0

    branch_top_k = top_k * 2  # each branch over-fetches for better fusion

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="search") as pool:
        fut_agentic   = pool.submit(_run_agentic,   query_bundle, branch_top_k)
        fut_heuristic = pool.submit(_run_heuristic, query_bundle, branch_top_k)

        # Collect results as they complete (order may vary)
        for fut in as_completed([fut_agentic, fut_heuristic]):
            if fut is fut_agentic:
                agentic_results, agentic_ms = fut.result()
            else:
                heuristic_results, heuristic_ms = fut.result()

    # ── 2. Cross-source RRF rerank ────────────────────────────────────────────
    t_rerank = time.perf_counter()
    final_candidates = cross_source_rerank(
        agentic_results=agentic_results,
        heuristic_results=heuristic_results,
        top_k=top_k,
    )
    rerank_ms = (time.perf_counter() - t_rerank) * 1000

    # ── 3. Build response ─────────────────────────────────────────────────────
    total_ms = (time.perf_counter() - t_start) * 1000
    raw_query = query_bundle.get("raw", query_bundle.get("cleaned", ""))

    response = build_response(
        query=raw_query,
        candidates=final_candidates,
        latency_ms={
            "agentic_ms":   round(agentic_ms,   2),
            "heuristic_ms": round(heuristic_ms, 2),
            "rerank_ms":    round(rerank_ms,    2),
            "total_ms":     round(total_ms,     2),
        },
    )

    logger.info(
        "Search done — agentic=%d heuristic=%d final=%d "
        "agentic=%.0fms heuristic=%.0fms wall=%.0fms",
        len(agentic_results), len(heuristic_results), len(final_candidates),
        agentic_ms, heuristic_ms, total_ms,
    )

    return response

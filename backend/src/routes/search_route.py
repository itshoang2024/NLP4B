"""
search_route.py — POST /search endpoint.

Receives raw query from frontend, passes through middleware for
cleaning/translation, then delegates to the search controller.
"""

from fastapi import APIRouter, Depends

from src.schemas import SearchRequest, SearchResponse
from src.middlewares.search_middleware import (
    clean_and_translate_middleware,
    ProcessedSearchRequest,
)
from src.controllers.search_controller import (
    execute_search,
    execute_agentic_only_search,
    execute_heuristic_only_search
)


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: ProcessedSearchRequest = Depends(clean_and_translate_middleware),
):
    """
    Unified search endpoint.

    Flow:
    1. Middleware cleans and translates the raw query → query_bundle
    2. Controller calls agentic + heuristic services
    3. Cross-source RRF rerank
    4. Response with azure URLs

    Request body (before middleware):
        {"raw_query": "...", "top_k": 10}

    Response:
        {"query": "...", "total_results": N, "results": [...], "latency_ms": {...}}
    """
    return execute_search(
        query_bundle=request.query_bundle,
        top_k=request.top_k,
        strategy=request.strategy,
    )


@router.post("/search/agentic", response_model=SearchResponse)
def search_agentic_endpoint(
    request: ProcessedSearchRequest = Depends(clean_and_translate_middleware),
):
    """
    Agentic-only search endpoint.
    
    Returns results from the intent-aware multimodal pipeline only,
    bypassing heuristic retrieval and RRF reranking.
    """
    return execute_agentic_only_search(
        query_bundle=request.query_bundle,
        top_k=request.top_k,
    )


@router.post("/search/heuristic", response_model=SearchResponse)
def search_heuristic_endpoint(
    request: ProcessedSearchRequest = Depends(clean_and_translate_middleware),
):
    """
    Heuristic-only search endpoint.
    
    Returns results from the dense hybrid pipeline only,
    bypassing agentic retrieval and RRF reranking.
    """
    return execute_heuristic_only_search(
        query_bundle=request.query_bundle,
        top_k=request.top_k,
    )
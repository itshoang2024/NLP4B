"""
api.py — FastAPI server for heuristic hybrid video-keyframe retrieval.

Endpoints:
  GET  /health          → system status
  POST /search          → hybrid RRF search, returns top-K keyframes

Run:
  cd retrieval/heuristic_retrieval
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import sys
import os

# ── Make sure sibling modules are importable when running directly ────────────
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import COLLECTION_NAME, DEFAULT_TOP_K
from models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    KeyframePayload,
    HealthResponse,
)
from embedder import encode_visual, encode_semantic, DEVICE
from searcher import hybrid_rrf_search


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LookUp.ai — Heuristic Retrieval API",
    description=(
        "Hybrid dense-only search over video keyframes using "
        "SigLIP (visual) + BGE-M3 (semantic) vectors with Qdrant RRF fusion."
    ),
    version="1.0.0",
)

# Allow all origins so the Streamlit front-end (any port) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Quick health check.
    Returns the collection name and compute device in use.
    Does NOT load models — stays fast.
    """
    return HealthResponse(
        status="ok",
        collection=COLLECTION_NAME,
        device=DEVICE,
    )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search(request: SearchRequest):
    """
    Hybrid keyframe search.

    Flow:
      1. Encode `query` → SigLIP 1152-d vector  (visual semantics)
      2. Encode `query` → BGE-M3 1024-d vector  (text semantics)
      3. Qdrant Prefetch × 2  →  RRF fusion  →  top-K results
      4. Return structured JSON ready for the UI.

    Request body:
      - query          : natural language search string
      - top_k          : how many results to return (1–50, default 10)
      - prefetch_limit : candidate pool per vector branch (10–200, default 50)
    """
    t_start = time.perf_counter()

    # ── Step 1: encode ────────────────────────────────────────────────────────
    try:
        t_enc_start = time.perf_counter()

        siglip_vec = encode_visual(request.query)
        bge_vec    = encode_semantic(request.query)

        encode_ms = (time.perf_counter() - t_enc_start) * 1000

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")

    # ── Step 2: search ────────────────────────────────────────────────────────
    try:
        t_search_start = time.perf_counter()

        raw_hits = hybrid_rrf_search(
            bge_vector=bge_vec,
            siglip_vector=siglip_vec,
            top_k=request.top_k,
            prefetch_limit=request.prefetch_limit,
        )

        search_ms = (time.perf_counter() - t_search_start) * 1000

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant search failed: {exc}")

    # ── Step 3: shape response ────────────────────────────────────────────────
    results = []
    for rank, hit in enumerate(raw_hits, start=1):
        p = hit["payload"]
        results.append(
            SearchResult(
                rank=rank,
                point_id=hit["point_id"],
                rrf_score=hit["rrf_score"],
                payload=KeyframePayload(
                    video_id=p.get("video_id"),
                    frame_idx=p.get("frame_idx"),
                    timestamp_sec=p.get("timestamp_sec"),
                    caption=p.get("caption"),
                    detailed_caption=p.get("detailed_caption"),
                    ocr_text=p.get("ocr_text"),
                    azure_url=p.get("azure_url"),
                    youtube_link=p.get("youtube_link"),
                ),
            )
        )

    total_ms = (time.perf_counter() - t_start) * 1000

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
        latency_ms={
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "total_ms":  round(total_ms,  2),
        },
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

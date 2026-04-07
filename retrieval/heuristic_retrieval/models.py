"""
models.py — Pydantic schemas for API request / response.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    prefetch_limit: int = Field(
        default=50, ge=10, le=200,
        description="Candidate pool size per vector before RRF fusion"
    )


# ── Per-result payload ────────────────────────────────────────────────────────

class KeyframePayload(BaseModel):
    video_id:        Optional[str]   = None
    frame_idx:       Optional[int]   = None
    timestamp_sec:   Optional[float] = None
    caption:         Optional[str]   = None
    detailed_caption: Optional[str]  = None
    ocr_text:        Optional[str]   = None
    azure_url:       Optional[str]   = None   # keyframe image URL
    youtube_link:    Optional[str]   = None   # deep-linked YouTube URL


class SearchResult(BaseModel):
    rank:       int
    point_id:   str
    rrf_score:  float
    payload:    KeyframePayload


# ── Response ──────────────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    query:          str
    total_results:  int
    results:        List[SearchResult]
    latency_ms:     Dict[str, float]   # encode_ms, search_ms, total_ms


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:     str
    collection: str
    device:     str

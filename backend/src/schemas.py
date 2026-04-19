"""
schemas.py — Shared Pydantic models and TypedDicts for the backend API.

API-facing models (Pydantic) and internal pipeline types (TypedDict)
are co-located here so that all layers share the same definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── API Request / Response ────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Raw search request from the frontend."""
    raw_query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    strategy: Literal["agentic", "heuristic", "both"] = Field(
        default="both",
        description="Search strategy: 'agentic' (intent-aware), 'heuristic' (dense hybrid), or 'both' (RRF fusion)",
    )


class SearchResultItem(BaseModel):
    """Single result in the search response (flat structure)."""
    rank: int
    video_id: str
    frame_id: int
    score: float
    branch: str                             # "agentic" | "heuristic" | "fused"
    azure_url: Optional[str] = None
    youtube_link: Optional[str] = None
    ocr_text: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Full response returned by POST /search."""
    query: str
    total_results: int
    results: List[SearchResultItem]
    latency_ms: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    collection: str
    device: str


# ── Internal pipeline types (TypedDict) ──────────────────────────────────────

from typing import TypedDict


class QueryBundle(TypedDict, total=False):
    raw: str
    cleaned: str
    lang: str
    translated_en: str
    rewrites: List[str]


class QueryIntent(TypedDict, total=False):
    objects: List[str]
    attributes: List[str]
    actions: List[str]
    scene: List[str]
    text_cues: List[str]
    metadata_cues: List[str]
    query_type: str


class Candidate(TypedDict, total=False):
    video_id: str
    frame_id: int
    score: float
    source: str
    branch: str
    fused_score: float
    rerank_score: float
    agent_score: float
    evidence: List[str]
    source_scores: Dict[str, float]
    raw_payload: Dict[str, Any]
    trace: Dict[str, Any]


class TraceLog(TypedDict, total=False):
    node: str
    payload: Dict[str, Any]

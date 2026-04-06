from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

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


class AgentState(TypedDict, total=False):
    raw_query: str

    query_bundle: QueryBundle
    query_intent: QueryIntent
    routing_weights: Dict[str, float]

    retrieval_results: Dict[str, List[Candidate]]
    fused_candidates: List[Candidate]
    reranked_candidates: List[Candidate]
    agent_topk: List[Candidate]

    trace_logs: List[TraceLog]
    error: Optional[str]
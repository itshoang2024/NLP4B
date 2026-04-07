"""
fusion node — merge multi-source candidates with weighted scores.

Migrated from: retrieval/agentic_retrieval/nodes/fusion.py
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from ..state import AgentState
from ..scoring import dedup_key, minmax_normalize


def fuse_candidates(retrieval_results: Dict[str, List[dict]], routing_weights: Dict[str, float]) -> List[dict]:
    normalized_by_source: Dict[str, List[dict]] = {}

    for source, items in retrieval_results.items():
        source_scores = [it["score"] for it in items]
        norm_scores = minmax_normalize(source_scores)

        normalized_items = []
        for it, ns in zip(items, norm_scores):
            copied = dict(it)
            copied["normalized_score"] = ns
            normalized_items.append(copied)
        normalized_by_source[source] = normalized_items

    merged: Dict[Tuple[str, int], dict] = {}

    for source, items in normalized_by_source.items():
        weight = routing_weights.get(source, 0.0)
        for it in items:
            key = dedup_key(it["video_id"], it["frame_id"])
            if key not in merged:
                merged[key] = {
                    "video_id": it["video_id"],
                    "frame_id": it["frame_id"],
                    "branch": "agentic",
                    "fused_score": 0.0,
                    "source_scores": {},
                    "evidence": [],
                    "raw_payload": {},
                }

            merged[key]["source_scores"][source] = it["normalized_score"]
            merged[key]["evidence"].append(source)
            merged[key]["fused_score"] += weight * it["normalized_score"]

    fused = list(merged.values())
    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused


def candidate_fusion_node(state: AgentState) -> AgentState:
    retrieval_results = state["retrieval_results"]
    routing_weights = state["routing_weights"]

    fused = fuse_candidates(retrieval_results, routing_weights)
    state["fused_candidates"] = fused

    state.setdefault("trace_logs", []).append({
        "node": "candidate_fusion",
        "payload": {
            "num_fused": len(fused),
            "top_5": fused[:5],
        },
    })
    return state

"""
rerank node — internal agentic multi-signal reranker.

Migrated from: retrieval/agentic_retrieval/nodes/rerank.py
This is the INTERNAL rerank within the agentic pipeline.
Cross-source reranking (agentic vs heuristic) is in controllers/rerank.py.
"""

from __future__ import annotations
from typing import Dict, List, Set

from ..state import AgentState


_INTENT_TO_EXPECTED_SOURCE: Dict[str, str] = {
    "text_cues": "ocr",
    "objects": "object",
}
_EVENT_SOURCES: Set[str] = {"caption", "keyframe"}


def _safe_list(d: dict, key: str) -> list:
    val = d.get(key, []) or []
    return [x for x in val if isinstance(x, str) and x.strip()]


def _expected_modalities(query_intent: dict) -> Set[str]:
    expected: Set[str] = set()
    for intent_key, source in _INTENT_TO_EXPECTED_SOURCE.items():
        if _safe_list(query_intent, intent_key):
            expected.add(source)
    if _safe_list(query_intent, "actions") or _safe_list(query_intent, "scene"):
        expected.add("caption")
    return expected


def rerank_frames(
    fused_candidates: List[dict],
    query_intent: dict,
    routing_weights: Dict[str, float],
    *,
    top_n_input: int = 30,
    alpha: float = 0.15,
    beta: float = 0.10,
    gamma: float = 0.08,
) -> List[dict]:
    """
    Multi-signal reranker:
        rerank_score = fused_score
                     + α · cross_source_agreement
                     + β · intent_coverage_bonus
                     - γ · missing_modality_penalty
    """
    shortlist = fused_candidates[:top_n_input]
    expected = _expected_modalities(query_intent)
    n_expected = max(len(expected), 1)

    reranked: List[dict] = []

    for item in shortlist:
        fused_score: float = item["fused_score"]
        evidence: List[str] = item.get("evidence", [])
        source_scores: Dict[str, float] = item.get("source_scores", {})

        agreement = sum(
            source_scores.get(src, 0.0) * routing_weights.get(src, 0.0)
            for src in evidence
        )
        agreement_bonus = alpha * agreement

        evidence_set = set(evidence)
        present = expected & evidence_set
        missing = expected - evidence_set

        coverage_bonus = beta * (len(present) / n_expected)
        missing_penalty = gamma * (len(missing) / n_expected)

        rerank_score = (
            fused_score
            + agreement_bonus
            + coverage_bonus
            - missing_penalty
        )

        copied = dict(item)
        copied["rerank_score"] = rerank_score
        copied["agent_score"] = rerank_score
        copied["score"] = rerank_score  # unified score field

        copied["rerank_signals"] = {
            "fused_score": round(fused_score, 5),
            "agreement_bonus": round(agreement_bonus, 5),
            "coverage_bonus": round(coverage_bonus, 5),
            "missing_penalty": round(missing_penalty, 5),
            "expected_modalities": sorted(expected),
            "present_modalities": sorted(present),
            "missing_modalities": sorted(missing),
        }

        reranked.append(copied)

    reranked.sort(key=lambda x: x["agent_score"], reverse=True)
    return reranked


def frame_reranking_node(state: AgentState) -> AgentState:
    query_intent = state["query_intent"]
    routing_weights = state["routing_weights"]
    fused_candidates = state["fused_candidates"]

    reranked = rerank_frames(
        fused_candidates=fused_candidates,
        query_intent=query_intent,
        routing_weights=routing_weights,
        top_n_input=30,
    )
    state["reranked_candidates"] = reranked
    state["agent_topk"] = reranked[:20]

    state.setdefault("trace_logs", []).append({
        "node": "frame_reranking",
        "payload": {
            "num_reranked": len(reranked),
            "top_5_signals": [
                {
                    "video_id": c.get("video_id"),
                    "frame_id": c.get("frame_id"),
                    "agent_score": round(c.get("agent_score", 0), 5),
                    **(c.get("rerank_signals", {})),
                }
                for c in state["agent_topk"][:5]
            ],
        },
    })
    return state

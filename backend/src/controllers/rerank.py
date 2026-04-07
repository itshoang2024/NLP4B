"""
rerank.py — Cross-source RRF reranking for merging agentic + heuristic results.

Uses Reciprocal Rank Fusion to merge candidates from two independent
retrieval branches without being biased by score scale differences.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def cross_source_rerank(
    agentic_results: List[Dict],
    heuristic_results: List[Dict],
    top_k: int = 10,
    *,
    rrf_k: int = 60,
) -> List[Dict]:
    """
    Merge results from agentic and heuristic branches using RRF.

    RRF formula per candidate:
        rrf_score = Σ  1 / (k + rank_in_branch)

    If the same (video_id, frame_id) appears in both branches,
    both rank contributions are summed → agreement bonus emerges naturally.

    Parameters
    ----------
    agentic_results : list[dict]
        Ranked candidates from the agentic pipeline.
    heuristic_results : list[dict]
        Ranked candidates from the heuristic pipeline.
    top_k : int
        How many final results to return.
    rrf_k : int
        RRF constant (default 60, standard value).

    Returns
    -------
    list[dict]
        Merged, deduplicated, re-scored candidates sorted by rrf_score.
    """
    # Track merged candidates by (video_id, frame_id)
    merged: Dict[Tuple[str, int], Dict] = {}

    def _process_branch(results: List[Dict], branch_name: str):
        for rank, item in enumerate(results, start=1):
            key = (item["video_id"], int(item["frame_id"]))
            rrf_contribution = 1.0 / (rrf_k + rank)

            if key not in merged:
                merged[key] = {
                    "video_id": item["video_id"],
                    "frame_id": int(item["frame_id"]),
                    "score": 0.0,
                    "branch": branch_name,
                    "evidence": list(item.get("evidence", [])),
                    "raw_payload": item.get("raw_payload", {}),
                    "source_scores": dict(item.get("source_scores", {})),
                    "branches": [branch_name],
                }
            else:
                existing = merged[key]
                if branch_name not in existing.get("branches", []):
                    existing["branches"].append(branch_name)
                    existing["branch"] = "fused"
                # Merge evidence lists
                for ev in item.get("evidence", []):
                    if ev not in existing["evidence"]:
                        existing["evidence"].append(ev)
                # Keep the richer payload
                if not existing.get("raw_payload") and item.get("raw_payload"):
                    existing["raw_payload"] = item["raw_payload"]

            merged[key]["score"] += rrf_contribution

    _process_branch(agentic_results, "agentic")
    _process_branch(heuristic_results, "heuristic")

    # Sort by RRF score descending
    final = list(merged.values())
    final.sort(key=lambda x: x["score"], reverse=True)

    # Clean up internal tracking field
    for item in final:
        item.pop("branches", None)

    return final[:top_k]

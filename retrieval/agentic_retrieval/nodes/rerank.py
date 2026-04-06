from __future__ import annotations

from state import AgentState


def rerank_frames(query_bundle: dict, fused_candidates: list[dict], top_n_input: int = 30) -> list[dict]:
    """
    Stub reranker:
    - hiện tại dùng fused_score
    - cộng nhẹ bonus nếu frame có evidence từ nhiều nguồn
    Sau này thay bằng VLM hoặc cross-modal reranker.
    """
    shortlist = fused_candidates[:top_n_input]
    reranked = []

    for item in shortlist:
        evidence_bonus = 0.03 * len(item.get("evidence", []))
        rerank_score = item["fused_score"] + evidence_bonus
        copied = dict(item)
        copied["rerank_score"] = rerank_score
        copied["agent_score"] = rerank_score
        reranked.append(copied)

    reranked.sort(key=lambda x: x["agent_score"], reverse=True)
    return reranked


def frame_reranking_node(state: AgentState) -> AgentState:
    query_bundle = state["query_bundle"]
    fused_candidates = state["fused_candidates"]

    reranked = rerank_frames(query_bundle, fused_candidates, top_n_input=30)
    state["reranked_candidates"] = reranked
    state["agent_topk"] = reranked[:20]

    state.setdefault("trace_logs", []).append({
        "node": "frame_reranking",
        "payload": {
            "num_reranked": len(reranked),
            "top_5": state["agent_topk"][:5],
        },
    })
    return state
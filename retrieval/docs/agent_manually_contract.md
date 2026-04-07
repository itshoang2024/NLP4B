# Cross-Branch Output Schema Contract

> **Post-refactor note:** This contract was written before the backend refactor. The canonical schema is now defined in `backend/src/schemas.py` (`SearchResultItem` for the API response) and the internal `Candidate` TypedDict. See [backend/README.md](../../backend/README.md) for the full API contract.

## 1. Internal candidate schema (used by both branches)

Each branch (`agentic_retrieve`, `heuristic_retrieve`) must return candidates as `list[dict]` with these keys:

```json
{
    "video_id": "str — YouTube video ID",
    "frame_id": "int — keyframe index",
    "score": "float — branch-internal score (higher = better)",
    "branch": "str — 'agentic' or 'heuristic'",
    "evidence": "list[str] — retrieval sources that matched",
    "raw_payload": "dict — Qdrant payload (azure_url, caption, etc.)"
}
```

Optional keys (agentic only): `agent_score`, `source_scores`, `rerank_signals`, `trace`.

## 2. Score direction

- Score must be **higher = better**
- Scores are **not** normalized across branches — the cross-source RRF reranker (`backend/src/controllers/rerank.py`) is scale-independent
- Final API response uses RRF-derived `score`, not raw branch scores

## 3. Frame identity

Both branches must use the same identity key:
- `video_id` (str) — YouTube video ID
- `frame_id` (int) — Qdrant point's `frame_idx` payload field

The RRF reranker deduplicates by `(video_id, frame_id)` tuple. Mismatched identity keys will prevent cross-branch agreement detection.
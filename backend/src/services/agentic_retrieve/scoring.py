from __future__ import annotations
from typing import Dict, List, Tuple


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-8:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def dedup_key(video_id: str, frame_id: int) -> Tuple[str, int]:
    return (video_id, frame_id)

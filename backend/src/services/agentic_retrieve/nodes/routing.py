"""
routing node — compute modality weights based on query intent.

Migrated from: retrieval/agentic_retrieval/nodes/routing.py
"""

from __future__ import annotations

from ..state import AgentState
from ..scoring import normalize_weights


QUERY_TYPE_PROFILES: dict[str, dict[str, float]] = {
    "mixed": {
        "keyframe": 0.25, "ocr": 0.10, "object": 0.25,
        "metadata": 0.00, "caption": 0.40,
    },
    "visual_object": {
        "keyframe": 0.25, "ocr": 0.05, "object": 0.50,
        "metadata": 0.00, "caption": 0.20,
    },
    "visual_event": {
        "keyframe": 0.25, "ocr": 0.05, "object": 0.15,
        "metadata": 0.00, "caption": 0.55,
    },
    "text_in_image": {
        "keyframe": 0.10, "ocr": 0.70, "object": 0.05,
        "metadata": 0.00, "caption": 0.15,
    },
    "metadata_hint": {
        "keyframe": 0.35, "ocr": 0.05, "object": 0.10,
        "metadata": 0.00, "caption": 0.50,
    },
}

EMPTY_INTENT_FALLBACK: dict[str, float] = {
    "keyframe": 0.40, "ocr": 0.00, "object": 0.10,
    "metadata": 0.00, "caption": 0.50,
}


def _safe_list(intent: dict, key: str) -> list[str]:
    value = intent.get(key, []) or []
    return [x for x in value if isinstance(x, str) and x.strip()]


def _is_intent_empty(intent: dict) -> bool:
    return not any(
        _safe_list(intent, key)
        for key in ("objects", "attributes", "actions", "scene", "text_cues", "metadata_cues")
    )


def _base_profile(intent: dict) -> dict[str, float]:
    query_type = intent.get("query_type", "mixed")
    return dict(QUERY_TYPE_PROFILES.get(query_type, QUERY_TYPE_PROFILES["mixed"]))


def _apply_semantic_adjustments(weights: dict[str, float], intent: dict) -> dict[str, float]:
    text_cues = _safe_list(intent, "text_cues")
    objects = _safe_list(intent, "objects")
    attributes = _safe_list(intent, "attributes")
    actions = _safe_list(intent, "actions")
    scene = _safe_list(intent, "scene")
    metadata_cues = _safe_list(intent, "metadata_cues")
    query_type = intent.get("query_type", "mixed")

    weights["metadata"] = 0.0

    if text_cues:
        weights["ocr"] += 0.25
        if len(text_cues) >= 2:
            weights["ocr"] += 0.10
        weights["caption"] -= 0.08
        weights["keyframe"] -= 0.05
    elif query_type != "text_in_image":
        weights["ocr"] = 0.0

    if objects:
        if len(objects) == 1:
            weights["object"] += 0.10
        elif len(objects) == 2:
            weights["object"] += 0.18
        else:
            weights["object"] += 0.25
        weights["caption"] -= 0.05
    elif query_type not in {"visual_object"}:
        weights["object"] = 0.0

    if actions:
        weights["caption"] += 0.12
    if scene:
        weights["caption"] += 0.10
        weights["keyframe"] += 0.05
    if actions and scene:
        weights["caption"] += 0.06

    if attributes and not text_cues:
        weights["keyframe"] += 0.08

    if metadata_cues:
        weights["caption"] += 0.08
        weights["keyframe"] += 0.04

    return weights


def compute_modality_weights(intent: dict) -> dict[str, float]:
    if _is_intent_empty(intent):
        return normalize_weights(dict(EMPTY_INTENT_FALLBACK))

    weights = _base_profile(intent)
    weights = _apply_semantic_adjustments(weights, intent)

    weights = {k: max(v, 0.0) for k, v in weights.items()}

    if sum(weights.values()) <= 0:
        return normalize_weights(dict(EMPTY_INTENT_FALLBACK))

    return normalize_weights(weights)


def modality_routing_node(state: AgentState) -> AgentState:
    intent = state["query_intent"]
    state["routing_weights"] = compute_modality_weights(intent)

    state.setdefault("trace_logs", []).append({
        "node": "modality_routing",
        "payload": {"routing_weights": state["routing_weights"]},
    })
    return state

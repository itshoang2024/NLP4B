from __future__ import annotations

from state import AgentState
from services.scoring import normalize_weights


QUERY_TYPE_PROFILES: dict[str, dict[str, float]] = {
    # Default balanced profile when intent is usable but not strongly specialized
    "mixed": {
        "keyframe": 0.25,
        "ocr": 0.10,
        "object": 0.25,
        "metadata": 0.00,
        "caption": 0.40,
    },
    # Query emphasizes visible objects / appearance
    "visual_object": {
        "keyframe": 0.25,
        "ocr": 0.05,
        "object": 0.50,
        "metadata": 0.00,
        "caption": 0.20,
    },
    # Query emphasizes actions / scene / event semantics
    "visual_event": {
        "keyframe": 0.25,
        "ocr": 0.05,
        "object": 0.15,
        "metadata": 0.00,
        "caption": 0.55,
    },
    # Query likely depends on words appearing in the frame
    "text_in_image": {
        "keyframe": 0.10,
        "ocr": 0.70,
        "object": 0.05,
        "metadata": 0.00,
        "caption": 0.15,
    },
    # Metadata is currently not usable, so fall back to visual/semantic evidence
    "metadata_hint": {
        "keyframe": 0.35,
        "ocr": 0.05,
        "object": 0.10,
        "metadata": 0.00,
        "caption": 0.50,
    },
}

# Fallback when intent extraction is empty or too poor to trust
EMPTY_INTENT_FALLBACK: dict[str, float] = {
    "keyframe": 0.40,
    "ocr": 0.00,
    "object": 0.10,
    "metadata": 0.00,
    "caption": 0.50,
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
    """
    Semi-rule-based adjustments:
    - text cues -> OCR up
    - concrete objects -> object up
    - actions / scene -> caption up
    - attributes-only visual descriptions -> keyframe mildly up
    - metadata currently disabled
    """
    text_cues = _safe_list(intent, "text_cues")
    objects = _safe_list(intent, "objects")
    attributes = _safe_list(intent, "attributes")
    actions = _safe_list(intent, "actions")
    scene = _safe_list(intent, "scene")
    metadata_cues = _safe_list(intent, "metadata_cues")
    query_type = intent.get("query_type", "mixed")

    # Metadata search is not usable yet -> force disable for now
    weights["metadata"] = 0.0

    # OCR
    if text_cues:
        weights["ocr"] += 0.25
        if len(text_cues) >= 2:
            weights["ocr"] += 0.10
        weights["caption"] -= 0.08
        weights["keyframe"] -= 0.05
    elif query_type != "text_in_image":
        weights["ocr"] = 0.0

    # Object-centric signals
    if objects:
        if len(objects) == 1:
            weights["object"] += 0.10
        elif len(objects) == 2:
            weights["object"] += 0.18
        else:
            weights["object"] += 0.25
        weights["caption"] -= 0.05
    elif query_type not in {"visual_object"}:
        # Turn off noisy object retrieval when query has no concrete object cues
        weights["object"] = 0.0

    # Event / scene understanding
    if actions:
        weights["caption"] += 0.12
    if scene:
        weights["caption"] += 0.10
        weights["keyframe"] += 0.05
    if actions and scene:
        weights["caption"] += 0.06

    # Visual appearance-heavy query: useful for direct image similarity
    if attributes and not text_cues:
        weights["keyframe"] += 0.08

    # Metadata cues exist but modality is disabled for now.
    # Re-route that weight demand into caption/keyframe instead of metadata.
    if metadata_cues:
        weights["caption"] += 0.08
        weights["keyframe"] += 0.04

    return weights


def compute_modality_weights(intent: dict) -> dict[str, float]:
    if _is_intent_empty(intent):
        return normalize_weights(dict(EMPTY_INTENT_FALLBACK))

    weights = _base_profile(intent)
    weights = _apply_semantic_adjustments(weights, intent)

    # Clamp negatives before normalization
    weights = {k: max(v, 0.0) for k, v in weights.items()}

    # If all non-metadata modalities somehow collapse to zero, recover gracefully
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

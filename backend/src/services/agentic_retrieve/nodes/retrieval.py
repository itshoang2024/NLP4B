"""
retrieval node — parallel multi-modal search against Qdrant.

Migrated from: retrieval/agentic_retrieval/nodes/retrieval.py
"""

from __future__ import annotations
from typing import List

from ..state import AgentState
from ..qdrant_search import QdrantSearchService


def build_query_texts(query_bundle: dict, query_intent: dict) -> List[str]:
    texts = []

    cleaned = query_bundle.get("cleaned", "")
    translated_en = query_bundle.get("translated_en", "")
    rewrites = query_bundle.get("rewrites", []) or []

    if cleaned:
        texts.append(cleaned)
    if translated_en and translated_en not in texts:
        texts.append(translated_en)

    for rw in rewrites:
        if rw and rw not in texts:
            texts.append(rw)

    objects = query_intent.get("objects", []) or []
    attributes = query_intent.get("attributes", []) or []
    actions = query_intent.get("actions", []) or []
    scene = query_intent.get("scene", []) or []
    text_cues = query_intent.get("text_cues", []) or []

    compact_parts = objects + attributes + actions + scene + text_cues
    compact_query = ", ".join(compact_parts).strip(", ")
    if compact_query and compact_query not in texts:
        texts.append(compact_query)

    return texts[:5]


def build_ocr_query_texts(query_intent: dict) -> List[str]:
    """Build OCR-specific query texts using ONLY the text_cues from intent."""
    text_cues = query_intent.get("text_cues", []) or []
    texts = [cue.strip() for cue in text_cues if isinstance(cue, str) and cue.strip()]
    return texts[:5]


def _adapt_results(raw_results: list[dict], source: str) -> list[dict]:
    adapted = []
    for item in raw_results:
        adapted.append({
            "video_id": item["video_id"],
            "frame_id": int(item["frame_id"]),
            "score": float(item["score"]),
            "source": source,
            "branch": "agentic",
            "raw_payload": item.get("raw_payload", {}),
        })
    return adapted


def parallel_retrieval_node_factory(search_service: QdrantSearchService, top_k_per_source: int = 20):
    def parallel_retrieval_node(state: AgentState) -> AgentState:
        qb = state["query_bundle"]
        qi = state["query_intent"]

        query_texts = build_query_texts(qb, qi)

        keyframe_results = _adapt_results(
            search_service.search_keyframe(query_texts, top_k=top_k_per_source),
            source="keyframe",
        )
        ocr_query_texts = build_ocr_query_texts(qi)
        ocr_results = _adapt_results(
            search_service.search_ocr(ocr_query_texts, top_k=top_k_per_source),
            source="ocr",
        )
        object_results = _adapt_results(
            search_service.search_object(query_texts, top_k=top_k_per_source),
            source="object",
        )
        metadata_results = _adapt_results(
            search_service.search_metadata(query_texts, top_k=top_k_per_source),
            source="metadata",
        )
        caption_results = _adapt_results(
            search_service.search_caption(query_texts, top_k=top_k_per_source),
            source="caption",
        )

        state["retrieval_results"] = {
            "keyframe": keyframe_results,
            "ocr": ocr_results,
            "object": object_results,
            "metadata": metadata_results,
            "caption": caption_results,
        }

        state.setdefault("trace_logs", []).append({
            "node": "parallel_retrieval",
            "payload": {
                "query_texts": query_texts,
                "ocr_query_texts": ocr_query_texts,
                "counts": {k: len(v) for k, v in state["retrieval_results"].items()},
            },
        })
        return state

    return parallel_retrieval_node

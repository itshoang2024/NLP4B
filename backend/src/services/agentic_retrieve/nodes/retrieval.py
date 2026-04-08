"""
retrieval node — optimized parallel multi-modal search against Qdrant.

v2 architecture (3-phase):
  Phase 1: Pre-embed — batch embed all query variants via /embed/query/batch
            (1-2 HTTP calls instead of 12+).
  Phase 2: Concurrent Qdrant — run 5 source searches in parallel threads
            using pre-computed vectors.
  Phase 3: Skip zero-weight — skip sources whose routing_weight == 0.

Old sequential approach (v1) called embed+search per-source per-variant,
resulting in ~12-15 sequential HTTP calls. The v2 approach reduces this to
~2 batch embed calls + ~5 concurrent Qdrant queries.

Migrated from: retrieval/agentic_retrieval/nodes/retrieval.py
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from ..state import AgentState
from ..qdrant_search import QdrantSearchService, PrecomputedEmbeddings


def _as_list(value) -> List[str]:
    """Coerce an intent field to a list of strings (LLMs sometimes return a bare string)."""
    if isinstance(value, list):
        return [str(x) for x in value if x]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


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

    objects = _as_list(query_intent.get("objects"))
    attributes = _as_list(query_intent.get("attributes"))
    actions = _as_list(query_intent.get("actions"))
    scene = _as_list(query_intent.get("scene"))
    text_cues = _as_list(query_intent.get("text_cues"))

    compact_parts = objects + attributes + actions + scene + text_cues
    compact_query = ", ".join(compact_parts).strip(", ")
    if compact_query and compact_query not in texts:
        texts.append(compact_query)

    return texts[:5]


def build_ocr_query_texts(query_intent: dict) -> List[str]:
    """Build OCR-specific query texts using ONLY the text_cues from intent."""
    text_cues = _as_list(query_intent.get("text_cues"))
    texts = [cue.strip() for cue in text_cues if cue.strip()]
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
    """Factory for the optimized parallel retrieval node.

    Architecture:
      Phase 1: Pre-embed all query variants + OCR texts via batch API
               (2 HTTP calls total instead of 12+).
      Phase 2: Run active Qdrant searches concurrently (ThreadPoolExecutor).
      Phase 3: Uses routing_weights from state to skip zero-weight sources.
    """

    def parallel_retrieval_node(state: AgentState) -> AgentState:
        qb = state["query_bundle"]
        qi = state["query_intent"]
        routing_weights: Dict[str, float] = state.get("routing_weights", {})

        query_texts = build_query_texts(qb, qi)
        ocr_query_texts = build_ocr_query_texts(qi)

        # Determine which sources to skip (weight == 0)
        active_sources = {
            k for k, v in routing_weights.items() if v > 0
        } if routing_weights else {"keyframe", "ocr", "object", "metadata", "caption"}

        # ── Phase 1: Pre-embed ────────────────────────────────────────
        # Batch embed main query texts + OCR texts separately
        # This replaces 12+ individual embed calls with 2 batch calls.
        main_embeddings: List[PrecomputedEmbeddings] = []
        ocr_embeddings: List[PrecomputedEmbeddings] = []

        # Only embed if at least one vector-based source is active
        vector_sources = {"keyframe", "caption", "object", "ocr"}
        if active_sources & vector_sources:
            main_embeddings, ocr_embeddings = search_service.embed_all_variants(
                query_texts=query_texts,
                ocr_texts=ocr_query_texts if "ocr" in active_sources else None,
            )

        # ── Phase 2: Concurrent Qdrant queries ────────────────────────
        # Each search uses pre-computed vectors → Qdrant-only, no embed calls.
        results: Dict[str, list] = {}

        # Build the search tasks — only for active sources
        search_tasks: Dict[str, tuple] = {}

        if "keyframe" in active_sources and main_embeddings:
            search_tasks["keyframe"] = (
                search_service.search_keyframe_with_vectors,
                (main_embeddings, top_k_per_source),
            )
        if "caption" in active_sources and main_embeddings:
            search_tasks["caption"] = (
                search_service.search_caption_with_vectors,
                (main_embeddings, top_k_per_source),
            )
        if "object" in active_sources and main_embeddings:
            search_tasks["object"] = (
                search_service.search_object_with_vectors,
                (main_embeddings, top_k_per_source),
            )
        if "ocr" in active_sources and ocr_embeddings:
            search_tasks["ocr"] = (
                search_service.search_ocr_with_vectors,
                (ocr_embeddings, top_k_per_source),
            )
        if "metadata" in active_sources and query_texts:
            search_tasks["metadata"] = (
                search_service.search_metadata,
                (query_texts, top_k_per_source),
            )

        # Execute all active searches concurrently
        if search_tasks:
            with ThreadPoolExecutor(
                max_workers=len(search_tasks),
                thread_name_prefix="qdrant",
            ) as pool:
                future_to_source = {
                    pool.submit(fn, *args): source
                    for source, (fn, args) in search_tasks.items()
                }
                for fut in as_completed(future_to_source):
                    source = future_to_source[fut]
                    try:
                        raw = fut.result()
                        results[source] = _adapt_results(raw, source)
                    except Exception as exc:
                        results[source] = []
                        import logging
                        logging.getLogger(__name__).warning(
                            "Qdrant search failed for source=%s: %s", source, exc,
                        )

        # Fill in empty lists for skipped sources
        for s in ("keyframe", "ocr", "object", "metadata", "caption"):
            results.setdefault(s, [])

        state["retrieval_results"] = results

        state.setdefault("trace_logs", []).append({
            "node": "parallel_retrieval",
            "payload": {
                "query_texts": query_texts,
                "ocr_query_texts": ocr_query_texts,
                "active_sources": sorted(active_sources),
                "skipped_sources": sorted(
                    {"keyframe", "ocr", "object", "metadata", "caption"} - active_sources
                ),
                "n_main_embeddings": len(main_embeddings),
                "n_ocr_embeddings": len(ocr_embeddings),
                "counts": {k: len(v) for k, v in results.items()},
            },
        })
        return state

    return parallel_retrieval_node

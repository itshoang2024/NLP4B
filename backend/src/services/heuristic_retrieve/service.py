"""
service.py — Heuristic Retrieval Service (Production)
======================================================

Pipeline:
  1. /embed/query → all 4 vectors + NLP analysis (1 HTTP call)
  2. execute_fallback_search():
       Tier 1 (prefilter): Qdrant filter by YOLO object tags → each stream top_k * 5
       Tier 2 (fallback):  remove filter if unique pool < top_k
       → Returns 4 independent ranked streams
  3. compute_rrf(): real per-stream ranking → accumulate 1/(k+rank)
  4. apply_count_bonus(): S_final = S_RRF * (1 + beta * M_avg)
  5. Sort + trim to top_k

Return dict keys: video_id, frame_id, score, branch, evidence, raw_payload
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# ── Vector names (aligned with keyframes_v1 collection) ──────────────────────
VEC_DENSE         = "keyframe-dense"
VEC_CAPTION_DENSE = "keyframe-caption-dense"
VEC_OBJECT_SPARSE = "keyframe-object-sparse"
VEC_OCR_SPARSE    = "keyframe-ocr-sparse"

COLLECTION_NAME = "keyframes_v1"

PAYLOAD_FIELDS = [
    "video_id", "frame_idx", "azure_url", "timestamp_sec",
    "timestamp_start", "timestamp_end",
    "youtube_link", "tags", "caption", "detailed_caption",
    "object_counts", "ocr_text", "title",
]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING API — /embed/query (gets all 4 vectors in 1 call)
# ══════════════════════════════════════════════════════════════════════════════

class EmbedQueryClient:
    def __init__(self, base_url: str, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            r = httpx.post(
                f"{self.base_url}/embed/query",
                json={"text": text.strip()},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("EmbedQueryClient.query failed: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# QDRANT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_candidate(point: Any, source: str) -> Optional[Dict[str, Any]]:
    payload = getattr(point, "payload", None) or {}
    video_id = payload.get("video_id")
    frame_idx = payload.get("frame_idx")
    if video_id is None or frame_idx is None:
        return None
    return {
        "video_id": str(video_id),
        "frame_id": int(frame_idx),
        "score": float(getattr(point, "score", 0.0)),  # raw Qdrant score (used for local rank)
        "source": source,
        "branch": "heuristic",
        "evidence": [source],
        "raw_payload": payload,
    }


def _extract_points(resp: Any) -> List[Any]:
    if resp is None:
        return []
    if hasattr(resp, "points"):
        return list(resp.points)
    if isinstance(resp, list):
        return resp
    return []


def _to_sparse(data: Dict[str, Any]) -> Optional[models.SparseVector]:
    idx, val = data.get("indices", []), data.get("values", [])
    if not idx:
        return None
    return models.SparseVector(indices=[int(i) for i in idx], values=[float(v) for v in val])


def _query_dense(
    client: QdrantClient, vector: List[float], using: str, source: str,
    limit: int, query_filter: Optional[models.Filter],
) -> List[Dict[str, Any]]:
    try:
        resp = client.query_points(
            collection_name=COLLECTION_NAME, query=vector, using=using,
            limit=limit, query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS, with_vectors=False,
        )
    except TypeError:
        resp = client.search(
            collection_name=COLLECTION_NAME, query_vector=(using, vector),
            limit=limit, query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS, with_vectors=False,
        )
    return [c for p in _extract_points(resp) if (c := _to_candidate(p, source))]


def _query_sparse(
    client: QdrantClient, sparse: models.SparseVector, using: str, source: str,
    limit: int, query_filter: Optional[models.Filter],
) -> List[Dict[str, Any]]:
    if not sparse.indices:
        return []
    try:
        resp = client.query_points(
            collection_name=COLLECTION_NAME, query=sparse, using=using,
            limit=limit, query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS, with_vectors=False,
        )
        return [c for p in _extract_points(resp) if (c := _to_candidate(p, source))]
    except Exception as exc:
        logger.warning("Sparse query failed (%s): %s", using, exc)
        return []


def _tag_filter(object_names: List[str]) -> Optional[models.Filter]:
    """Qdrant prefilter: only return frames that have at least one matching object tag."""
    if not object_names:
        return None
    return models.Filter(must=[
        models.FieldCondition(key="tags", match=models.MatchAny(any=object_names))
    ])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 2-tier fallback search
# ══════════════════════════════════════════════════════════════════════════════

def execute_fallback_search(
    client: QdrantClient,
    embed_resp: Dict[str, Any],
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    """
    Run 4 Qdrant searches, each fetching top_k * 5 candidates.

    The prefilter (Qdrant tag filter) reduces the index Qdrant must scan
    internally → saves CPU/RAM. It is transparent to RRF logic.

    Tier 1 (Strict prefilter):
        Filter by YOLO object tags. Each of 4 streams returns top_k * 5.
        If unique frames pooled across streams < top_k → Tier 2.

    Tier 2 (No filter):
        Remove the filter. object_sparse (nouns + synonyms from NLP) acts
        as soft-filter via BM25 score weighting.

    Returns
    -------
    list[list[dict]]
        4 independent streams, each sorted by Qdrant score (descending).
        RRF operates on these raw per-stream ranks.
    """
    nlp      = embed_resp.get("nlp_analysis", {})
    objects  = nlp.get("objects", [])
    obj_names = [o["object"] for o in objects]

    sem_vec = embed_resp["semantic_dense"]["embedding"]
    vis_vec = embed_resp["visual_dense"]["embedding"]
    obj_sp  = _to_sparse(embed_resp["object_sparse"])
    ocr_sp  = _to_sparse(embed_resp["ocr_sparse"])

    stream_limit = top_k * 5  # each stream fetches this many; pool ≈ 4 × stream_limit unique

    def _run(f: Optional[models.Filter]) -> List[List[Dict[str, Any]]]:
        streams = [
            _query_dense(client, sem_vec, VEC_CAPTION_DENSE, "caption",  stream_limit, f),
            _query_dense(client, vis_vec, VEC_DENSE,          "keyframe", stream_limit, f),
        ]
        if obj_sp:
            streams.append(_query_sparse(client, obj_sp, VEC_OBJECT_SPARSE, "object", stream_limit, f))
        if ocr_sp:
            streams.append(_query_sparse(client, ocr_sp, VEC_OCR_SPARSE,    "ocr",    stream_limit, f))
        return streams

    # Tier 1
    flt = _tag_filter(obj_names)
    streams = _run(flt)

    unique = {(c["video_id"], c["frame_id"]) for s in streams for c in s}
    logger.info("Tier-1 (filter=%r): %d unique frames", obj_names, len(unique))

    # Tier 2 — fallback if pool too small
    if len(unique) < top_k:
        logger.info("Tier-2 fallback: %d < %d, removing filter", len(unique), top_k)
        streams = _run(None)
        unique = {(c["video_id"], c["frame_id"]) for s in streams for c in s}
        logger.info("Tier-2: %d unique frames", len(unique))

    return streams


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — True Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def compute_rrf(
    streams: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    True RRF: rank each stream independently by Qdrant score, then
    accumulate 1/(rrf_k + rank) for every document in the pool.

        S_RRF(d) = Σ_stream  1 / (rrf_k + rank_of_d_in_stream)

    Documents absent from a stream contribute 0 for that stream.
    Documents appearing in multiple streams (especially both dense
    and sparse) get a natural agreement bonus.

    BM25 cannot dominate: each stream contributes at most 1/(60+1)
    regardless of its raw score magnitude.
    """
    pool: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for stream in streams:
        # Each stream is sorted by Qdrant score descending → rank = index + 1
        for rank_0, cand in enumerate(stream):
            key = (cand["video_id"], cand["frame_id"])
            contribution = 1.0 / (rrf_k + rank_0 + 1)

            if key in pool:
                pool[key]["score"] += contribution
                src = cand.get("source", "")
                if src and src not in pool[key]["evidence"]:
                    pool[key]["evidence"].append(src)
            else:
                pool[key] = {
                    **cand,
                    "score": contribution,
                    "evidence": [cand.get("source", "")],
                }

    result = list(pool.values())
    result.sort(key=lambda c: c["score"], reverse=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Count Bonus Multiplier (applied on top of RRF)
# ══════════════════════════════════════════════════════════════════════════════

def apply_count_bonus(
    rrf_pool: List[Dict[str, Any]],
    nlp_analysis: Dict[str, Any],
    top_k: int,
    beta: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Apply Count Bonus Multiplier on top of the RRF-ranked pool.

    For each frame:
        M_i     = 1.0 / (1.0 + |C_act - C_req|)   ← match quality per object
        M_avg   = mean(M_i)                          ← overall count alignment
        S_final = S_RRF * (1.0 + beta * M_avg)      ← multiplier boost

    Multiplier (not additive) keeps the score distribution intact.
    When there are no count requirements, M_avg = 0 → S_final = S_RRF.

    Parameters
    ----------
    rrf_pool  : list sorted by S_RRF descending (output of compute_rrf)
    nlp_analysis : NLP analysis from /embed/query
    top_k     : final number of results
    beta      : count bonus weight (0.15 = 15% max boost for perfect match)
    """
    count_objects = [o for o in nlp_analysis.get("objects", []) if o.get("count") is not None]

    for cand in rrf_pool:
        if count_objects:
            frame_counts: Dict[str, int] = (
                cand.get("raw_payload", {}).get("object_counts") or {}
            )
            m_values = []
            for obj in count_objects:
                c_req = obj["count"]
                c_act = frame_counts.get(obj["object"], 0)
                delta = abs(c_act - c_req)
                # M_i = 1.0 when delta=0 (perfect), approaches 0 as delta → ∞
                m_values.append(1.0 / (1.0 + delta))
            m_avg = sum(m_values) / len(m_values)
        else:
            m_avg = 0.0

        # S_final = S_RRF × (1 + β × M_avg)
        cand["score"] = round(cand["score"] * (1.0 + beta * m_avg), 8)

    rrf_pool.sort(key=lambda c: c["score"], reverse=True)
    return rrf_pool[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class HeuristicRetrieveService:
    """
    Production heuristic retrieval.

    Usage (unchanged from mock interface):
        service = HeuristicRetrieveService()
        candidates = service.retrieve(query_bundle, top_k=10)
    """

    def __init__(self) -> None:
        from src.config import get_qdrant_url, get_qdrant_api_key, get_embedding_api_url
        self._qdrant = QdrantClient(
            url=get_qdrant_url(), api_key=get_qdrant_api_key(), timeout=60
        )
        self._embed = EmbedQueryClient(base_url=get_embedding_api_url(), timeout=90)
        logger.info("HeuristicRetrieveService initialized (production)")

    def retrieve(
        self,
        query_bundle: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: embed → 2-tier search → RRF → count bonus.

        Parameters
        ----------
        query_bundle : dict  (translated_en > cleaned > raw_query)
        top_k        : number of final results

        Returns
        -------
        list[dict] with keys: video_id, frame_id, score, branch, evidence, raw_payload
        """
        query_text = (
            query_bundle.get("translated_en")
            or query_bundle.get("cleaned")
            or query_bundle.get("raw_query", "")
        ).strip()

        if not query_text:
            logger.warning("retrieve: empty query")
            return []

        logger.info("retrieve: query=%r top_k=%d", query_text, top_k)

        # 1. /embed/query → all 4 vectors + NLP
        embed_resp = self._embed.query(query_text)
        if embed_resp is None:
            logger.error("Embedding API unavailable")
            return []

        nlp_analysis = embed_resp.get("nlp_analysis", {})
        logger.info(
            "NLP: objects=%s counts=%s ocr=%s",
            [o["object"] for o in nlp_analysis.get("objects", [])],
            nlp_analysis.get("object_counts", {}),
            nlp_analysis.get("ocr_texts", []),
        )

        # 2. 2-tier search → 4 independent streams (each top_k * 5)
        streams = execute_fallback_search(self._qdrant, embed_resp, top_k)

        if not any(streams):
            logger.warning("All Qdrant streams returned empty")
            return []

        # 3. True RRF — rank each stream independently, accumulate scores
        rrf_pool = compute_rrf(streams)

        # 4. Count bonus multiplier → trim to top_k
        ranked = apply_count_bonus(rrf_pool, nlp_analysis, top_k)

        logger.info("retrieve: returning %d results", len(ranked))
        return ranked

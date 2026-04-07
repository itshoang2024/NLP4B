"""
searcher.py — Qdrant native hybrid search using Prefetch + RRF Fusion.

Architecture:
  Prefetch(SigLIP  → keyframe-dense)
  Prefetch(BGE-M3  → keyframe-caption-dense)
  └─► FusionQuery(Fusion.RRF)  →  Top-K
"""

from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

from config import COLLECTION_NAME, DEFAULT_TOP_K, DEFAULT_PREFETCH, get_qdrant_url, get_qdrant_api_key


# ── Singleton Qdrant client ───────────────────────────────────────────────────

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a module-level singleton QdrantClient."""
    global _qdrant_client
    if _qdrant_client is None:
        url = get_qdrant_url()
        api_key = get_qdrant_api_key()
        print(f"[searcher] Connecting to Qdrant at {url[:40]}...")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
        print("[searcher] Qdrant client ready.")
    return _qdrant_client


# ── Core search function ──────────────────────────────────────────────────────

def hybrid_rrf_search(
    bge_vector:     list,
    siglip_vector:  list,
    top_k:          int = DEFAULT_TOP_K,
    prefetch_limit: int = DEFAULT_PREFETCH,
    collection:     str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Execute Qdrant-native hybrid search with RRF fusion.

    Steps:
      1. Prefetch top-{prefetch_limit} from 'keyframe-caption-dense' (BGE-M3)
      2. Prefetch top-{prefetch_limit} from 'keyframe-dense'         (SigLIP)
      3. Merge candidates via Reciprocal Rank Fusion → return top-{top_k}

    Returns a list of dicts:
      { point_id, rrf_score, payload }
    """
    client = get_qdrant_client()

    hits = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=bge_vector,
                using="keyframe-caption-dense",
                limit=prefetch_limit,
            ),
            Prefetch(
                query=siglip_vector,
                using="keyframe-dense",
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "point_id": str(hit.id),
            "rrf_score": hit.score,
            "payload":   hit.payload or {},
        }
        for hit in hits
    ]

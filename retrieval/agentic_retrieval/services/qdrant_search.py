from __future__ import annotations

"""
qdrant_search.py — Qdrant-backed retrieval service for the Agentic Retrieval Branch.

Current collection design assumptions (aligned with qdrant_upsert.py):
- Single collection: keyframes_v1
- Named vectors in the same collection:
    * keyframe-dense           (SigLIP 1152d)
    * keyframe-caption-dense   (BGE-M3 1024d)
    * keyframe-object-sparse   (BM25 sparse)
    * keyframe-ocr-sparse      (BM25 sparse)
- Core payload fields per point:
    * video_id, frame_idx, azure_url, timestamp_sec, youtube_link
    * optional: tags, caption, detailed_caption, object_counts, ocr_text

Embedding behavior:
- This module no longer loads local embedding models.
- It calls the Azure Embedding API service for:
    * /embed/visual   (SigLIP text vector)
    * /embed/semantic (BGE-M3 text vector)
    * /embed/sparse   (BM25 sparse vector)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from qdrant_client import QdrantClient, models


# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[QdrantSearch] %(levelname)s: %(message)s")


# ── constants aligned with qdrant_upsert.py ───────────────────────────────────
COLLECTION_NAME = "keyframes_v1"

VEC_DENSE = "keyframe-dense"
VEC_CAPTION_DENSE = "keyframe-caption-dense"
VEC_OBJECT_SPARSE = "keyframe-object-sparse"
VEC_OCR_SPARSE = "keyframe-ocr-sparse"

DEFAULT_EMBEDDING_API_BASE_URL = "http://localhost:8000"
EMBED_VISUAL_PATH = "/embed/visual"
EMBED_SEMANTIC_PATH = "/embed/semantic"
EMBED_SPARSE_PATH = "/embed/sparse"

DEFAULT_PAYLOAD_FIELDS = [
    "video_id",
    "frame_idx",
    "azure_url",
    "timestamp_sec",
    "youtube_link",
    "tags",
    "caption",
    "detailed_caption",
    "object_counts",
    "ocr_text",
    # Keep a few possible title fields here in case the collection has already been patched.
    "title",
    "video_title",
    "youtube_title",
]

POSSIBLE_TITLE_FIELDS = ("title", "video_title", "youtube_title")


# ── text helpers ──────────────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


# ── embedding api client ──────────────────────────────────────────────────────
class EmbeddingApiClient:
    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = (base_url or DEFAULT_EMBEDDING_API_BASE_URL).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, text: str) -> Optional[Dict[str, Any]]:
        payload = {"text": _normalize_text(text)}
        if not payload["text"]:
            return None

        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.warning("Embedding API returned non-object JSON for %s", path)
                return None
            return data
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding API call failed for %s: %s", path, exc)
            return None

    def encode_semantic(self, text: str) -> Optional[List[float]]:
        data = self._post(EMBED_SEMANTIC_PATH, text)
        if not data:
            return None
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            logger.warning("Invalid semantic embedding response shape")
            return None
        return [float(x) for x in embedding]

    def encode_visual(self, text: str) -> Optional[List[float]]:
        data = self._post(EMBED_VISUAL_PATH, text)
        if not data:
            return None
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            logger.warning("Invalid visual embedding response shape")
            return None
        return [float(x) for x in embedding]

    def encode_sparse(self, text: str) -> Optional[models.SparseVector]:
        data = self._post(EMBED_SPARSE_PATH, text)
        if not data:
            return None

        indices = data.get("indices")
        values = data.get("values")
        if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
            logger.warning("Invalid sparse embedding response shape")
            return None

        try:
            return models.SparseVector(
                indices=[int(i) for i in indices],
                values=[float(v) for v in values],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse sparse embedding response: %s", exc)
            return None


# ── result helpers ────────────────────────────────────────────────────────────
def _extract_points(response: Any) -> List[Any]:
    """
    Compatible with both:
    - QueryResponse(points=[...]) from query_points()
    - plain list[ScoredPoint] from older client methods
    """
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if hasattr(response, "points"):
        return list(response.points)
    return []


def _payload_to_result(point: Any, source: str) -> Optional[Dict[str, Any]]:
    payload = getattr(point, "payload", None) or {}

    video_id = payload.get("video_id")
    frame_idx = payload.get("frame_idx")
    if video_id is None or frame_idx is None:
        return None

    return {
        "video_id": str(video_id),
        "frame_id": int(frame_idx),
        "score": float(getattr(point, "score", 0.0)),
        "source": source,
        "raw_payload": payload,
    }


def _merge_variant_results(results: Iterable[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
    """Merge hits from multiple query variants, keeping the best score per frame."""
    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for batch in results:
        for item in batch:
            key = (item["video_id"], int(item["frame_id"]))
            old = merged.get(key)
            if old is None or item["score"] > old["score"]:
                merged[key] = item

    final = list(merged.values())
    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:top_k]


# ── service ───────────────────────────────────────────────────────────────────
@dataclass
class QdrantSearchService:
    collection_name: str = COLLECTION_NAME
    url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
    payload_fields: Optional[List[str]] = None
    metadata_enabled: bool = True
    embedding_api_base_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.url = self.url or os.environ.get("QDRANT_URL")
        self.api_key = self.api_key or os.environ.get("QDRANT_API_KEY")
        if not self.url or not self.api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY.")

        embedding_api_base_url = (
            self.embedding_api_base_url
            or os.environ.get("EMBEDDING_API_BASE_URL")
            or DEFAULT_EMBEDDING_API_BASE_URL
        )

        self.embedding_client = EmbeddingApiClient(
            base_url=embedding_api_base_url,
            timeout=self.timeout,
        )

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
        )
        self.payload_fields = self.payload_fields or list(DEFAULT_PAYLOAD_FIELDS)
        self._warned_metadata = False

    # ── public API used by parallel_retrieval_node ──────────────────────────
    def search_keyframe(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against SigLIP visual vectors using text queries."""
        return self._search_dense_many(
            query_texts=query_texts,
            using=VEC_DENSE,
            encoder=self.embedding_client.encode_visual,
            source="keyframe",
            top_k=top_k,
        )

    def search_caption(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against BGE-M3 caption vectors using text queries."""
        return self._search_dense_many(
            query_texts=query_texts,
            using=VEC_CAPTION_DENSE,
            encoder=self.embedding_client.encode_semantic,
            source="caption",
            top_k=top_k,
        )

    def search_object(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against sparse object vectors."""
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OBJECT_SPARSE,
            encoder=self.embedding_client.encode_sparse,
            source="object",
            top_k=top_k,
        )

    def search_ocr(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against sparse OCR vectors."""
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OCR_SPARSE,
            encoder=self.embedding_client.encode_sparse,
            source="ocr",
            top_k=top_k,
        )

    def search_metadata(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Safe current behavior:
        - if no title-like payload field exists in the collection, return []
        - if one exists, run a lexical MatchText filter as a lightweight fallback

        This is NOT semantic metadata retrieval. For that, add a dedicated metadata vector.
        """
        if not self.metadata_enabled:
            return []

        title_field = self._discover_title_field()
        if title_field is None:
            if not self._warned_metadata:
                logger.warning(
                    "Metadata retrieval is disabled because no title-like payload field was found in collection '%s'. "
                    "Add title/video_title/youtube_title payload or a dedicated metadata vector.",
                    self.collection_name,
                )
                self._warned_metadata = True
            return []

        query_texts = [_normalize_text(q) for q in query_texts if _normalize_text(q)]
        if not query_texts:
            return []

        batches: List[List[Dict[str, Any]]] = []
        for text in query_texts:
            try:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=title_field,
                            match=models.MatchText(text=text),
                        )
                    ]
                )
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=top_k,
                    with_payload=self.payload_fields,
                    with_vectors=False,
                )
                points = response[0] if isinstance(response, tuple) else response
                hits = []
                for rank, point in enumerate(points, start=1):
                    payload = getattr(point, "payload", None) or {}
                    video_id = payload.get("video_id")
                    frame_idx = payload.get("frame_idx")
                    if video_id is None or frame_idx is None:
                        continue
                    hits.append(
                        {
                            "video_id": str(video_id),
                            "frame_id": int(frame_idx),
                            "score": 1.0 / rank,
                            "source": "metadata",
                            "raw_payload": payload,
                        }
                    )
                batches.append(hits)
            except Exception as exc:  # pragma: no cover
                logger.warning("Metadata lexical retrieval failed for text=%r: %s", text, exc)

        return _merge_variant_results(batches, top_k=top_k)

    # ── internal search methods ──────────────────────────────────────────────
    def _search_dense_many(
        self,
        query_texts: List[str],
        using: str,
        encoder: Callable[[str], Optional[List[float]]],
        source: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        batches: List[List[Dict[str, Any]]] = []
        for text in query_texts:
            vector = encoder(text)
            if vector is None:
                continue
            hits = self._query_points(query=vector, using=using, source=source, limit=top_k)
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def _search_sparse_many(
        self,
        query_texts: List[str],
        using: str,
        encoder: Callable[[str], Optional[models.SparseVector]],
        source: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        batches: List[List[Dict[str, Any]]] = []
        for text in query_texts:
            sparse_query = encoder(text)
            if sparse_query is None:
                continue
            hits = self._query_points(query=sparse_query, using=using, source=source, limit=top_k)
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def _query_points(
        self,
        query: Any,
        using: str,
        source: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                using=using,
                limit=limit,
                with_payload=self.payload_fields,
                with_vectors=False,
            )
        except TypeError:
            # Some client versions may prefer search/query_vector style.
            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query,
                using=using,
                limit=limit,
                with_payload=self.payload_fields,
                with_vectors=False,
            )

        points = _extract_points(response)
        results: List[Dict[str, Any]] = []
        for point in points:
            item = _payload_to_result(point, source=source)
            if item is not None:
                results.append(item)
        return results

    def _discover_title_field(self) -> Optional[str]:
        """Inspect a payload-bearing point to detect whether title-like fields exist."""
        try:
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=list(POSSIBLE_TITLE_FIELDS),
                with_vectors=False,
            )
            points = response[0] if isinstance(response, tuple) else response
            if not points:
                return None
            payload = getattr(points[0], "payload", None) or {}
            for field in POSSIBLE_TITLE_FIELDS:
                value = payload.get(field)
                if isinstance(value, str) and value.strip():
                    return field
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to inspect metadata title field: %s", exc)
        return None

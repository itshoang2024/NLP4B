"""
qdrant_search.py — Qdrant-backed retrieval service for the Agentic Retrieval Branch.

Migrated from: retrieval/agentic_retrieval/services/qdrant_search.py
Import paths updated for backend package structure.

Collection design (aligned with qdrant_upsert.py):
- Single collection: keyframes_v1
- Named vectors:
    * keyframe-dense           (SigLIP 1152d)
    * keyframe-caption-dense   (BGE-M3 1024d)
    * keyframe-object-sparse   (BM25 sparse)
    * keyframe-ocr-sparse      (BM25 sparse)
- Core payload fields per point:
    * video_id, frame_idx, azure_url, timestamp_sec, youtube_link
    * optional: tags, caption, detailed_caption, object_counts, ocr_text

Embedding API usage:
  Two modes of operation:

  A) Inline embedding (LEGACY — kept for backward compat & tests):
     Each search_*() method calls embed API per query text:
       - /embed/visual   (SigLIP text vector)
       - /embed/semantic (BGE-M3 text vector)
       - /embed/sparse   (BM25 sparse vector)
     Pros: simple. Cons: N texts × M sources = N*M HTTP calls.

  B) Pre-computed vectors (OPTIMIZED — used by parallel_retrieval_node):
     1. embed_all_variants(query_texts, ocr_texts) calls /embed/query/batch
        once per batch → returns PrecomputedEmbeddings for all texts.
     2. search_*_with_vectors(embeddings, top_k) methods receive
        pre-computed vectors → only Qdrant queries, no embedding calls.
     Pros: 12+ HTTP calls → 2 batch calls. Cons: slightly more complex.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from qdrant_client import QdrantClient, models


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
EMBED_QUERY_PATH = "/embed/query"
EMBED_QUERY_BATCH_PATH = "/embed/query/batch"

DEFAULT_PAYLOAD_FIELDS = [
    "video_id",
    "frame_idx",
    "azure_url",
    "timestamp_sec",
    "timestamp_start",
    "timestamp_end",
    "youtube_link",
    "tags",
    "caption",
    "detailed_caption",
    "object_counts",
    "ocr_text",
    "title",
    "video_title",
    "youtube_title",
]

POSSIBLE_TITLE_FIELDS = ("title", "video_title", "youtube_title")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


# ── embedding api client ──────────────────────────────────────────────────────
# ── pre-computed embedding container ──────────────────────────────────────────
@dataclass
class PrecomputedEmbeddings:
    """All embedding vectors for a single query text, pre-computed via /embed/query/batch."""
    text: str
    semantic_dense: List[float] = field(default_factory=list)   # BGE-M3 1024d
    visual_dense: List[float] = field(default_factory=list)     # SigLIP 1152d
    object_sparse: Optional[models.SparseVector] = None
    ocr_sparse: Optional[models.SparseVector] = None


# ── embedding api client ──────────────────────────────────────────────────────
class EmbeddingApiClient:
    """HTTP client for the Embedding-as-a-Service API.

    Provides two usage patterns:
      1. Individual encode_* methods (legacy, used by search_* methods).
      2. embed_batch() — calls /embed/query/batch for multiple texts,
         returns PrecomputedEmbeddings. Used by the optimized retrieval node.
    """

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
        except Exception as exc:
            logger.warning("Embedding API call failed for %s: %s", path, exc)
            return None

    # ── individual encoders (legacy) ──────────────────────────────────
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
        except Exception as exc:
            logger.warning("Failed to parse sparse embedding response: %s", exc)
            return None

    # ── batch embed (optimized) ───────────────────────────────────────
    @staticmethod
    def _parse_sparse(data: Dict[str, Any]) -> Optional[models.SparseVector]:
        """Parse sparse vector from API response dict."""
        indices = data.get("indices", [])
        values = data.get("values", [])
        if not indices:
            return None
        try:
            return models.SparseVector(
                indices=[int(i) for i in indices],
                values=[float(v) for v in values],
            )
        except Exception:
            return None

    def embed_batch(self, texts: List[str]) -> List[PrecomputedEmbeddings]:
        """Call /embed/query/batch for multiple texts → PrecomputedEmbeddings.

        Falls back to per-text /embed/query if batch endpoint unavailable.
        """
        clean = [_normalize_text(t) for t in texts]
        clean = [t for t in clean if t]
        if not clean:
            return []

        # Try batch endpoint first
        try:
            resp = httpx.post(
                f"{self.base_url}{EMBED_QUERY_BATCH_PATH}",
                json={"texts": clean},
                timeout=self.timeout * 2,  # batch may take longer
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            results: List[PrecomputedEmbeddings] = []
            for item in items:
                results.append(PrecomputedEmbeddings(
                    text=item.get("text", ""),
                    semantic_dense=item.get("semantic_dense", {}).get("embedding", []),
                    visual_dense=item.get("visual_dense", {}).get("embedding", []),
                    object_sparse=self._parse_sparse(item.get("object_sparse", {})),
                    ocr_sparse=self._parse_sparse(item.get("ocr_sparse", {})),
                ))
            logger.info("embed_batch: %d texts → %d embeddings via batch endpoint", len(clean), len(results))
            return results
        except Exception as exc:
            logger.warning("Batch endpoint failed, falling back to per-text: %s", exc)

        # Fallback: call /embed/query per text
        results = []
        for text in clean:
            try:
                data = self._post(EMBED_QUERY_PATH, text)
                if data is None:
                    continue
                results.append(PrecomputedEmbeddings(
                    text=text,
                    semantic_dense=data.get("semantic_dense", {}).get("embedding", []),
                    visual_dense=data.get("visual_dense", {}).get("embedding", []),
                    object_sparse=self._parse_sparse(data.get("object_sparse", {})),
                    ocr_sparse=self._parse_sparse(data.get("ocr_sparse", {})),
                ))
            except Exception as text_exc:
                logger.warning("embed_query fallback failed for %r: %s", text, text_exc)
        logger.info("embed_batch fallback: %d texts → %d embeddings", len(clean), len(results))
        return results


# ── result helpers ────────────────────────────────────────────────────────────
def _extract_points(response: Any) -> List[Any]:
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

    # ── public API (legacy — embeds inline per text) ───────────────────
    def search_keyframe(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return self._search_dense_many(
            query_texts=query_texts,
            using=VEC_DENSE,
            encoder=self.embedding_client.encode_visual,
            source="keyframe",
            top_k=top_k,
        )

    def search_caption(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return self._search_dense_many(
            query_texts=query_texts,
            using=VEC_CAPTION_DENSE,
            encoder=self.embedding_client.encode_semantic,
            source="caption",
            top_k=top_k,
        )

    def search_object(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OBJECT_SPARSE,
            encoder=self.embedding_client.encode_sparse,
            source="object",
            top_k=top_k,
        )

    def search_ocr(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OCR_SPARSE,
            encoder=self.embedding_client.encode_sparse,
            source="ocr",
            top_k=top_k,
        )

    # ── public API (optimized — pre-computed vectors, Qdrant only) ────
    def embed_all_variants(
        self,
        query_texts: List[str],
        ocr_texts: Optional[List[str]] = None,
    ) -> Tuple[List[PrecomputedEmbeddings], List[PrecomputedEmbeddings]]:
        """Pre-embed all query variants + OCR texts via batch endpoint.

        Returns (main_embeddings, ocr_embeddings) where:
          - main_embeddings: embeddings for general query texts
            (used for keyframe, caption, object searches)
          - ocr_embeddings: embeddings for OCR-specific texts
            (used only for OCR sparse search)

        OCR texts are embedded separately because they typically contain
        only text_cues from intent extraction, not full query text.
        """
        main_embs = self.embedding_client.embed_batch(query_texts) if query_texts else []
        ocr_embs = self.embedding_client.embed_batch(ocr_texts) if ocr_texts else []
        return main_embs, ocr_embs

    def search_keyframe_with_vectors(
        self, embeddings: List[PrecomputedEmbeddings], top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Keyframe search using pre-computed visual_dense vectors (no embed call)."""
        batches = []
        for emb in embeddings:
            if not emb.visual_dense:
                continue
            hits = self._query_points(
                query=emb.visual_dense, using=VEC_DENSE,
                source="keyframe", limit=top_k,
            )
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def search_caption_with_vectors(
        self, embeddings: List[PrecomputedEmbeddings], top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Caption search using pre-computed semantic_dense vectors (no embed call)."""
        batches = []
        for emb in embeddings:
            if not emb.semantic_dense:
                continue
            hits = self._query_points(
                query=emb.semantic_dense, using=VEC_CAPTION_DENSE,
                source="caption", limit=top_k,
            )
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def search_object_with_vectors(
        self, embeddings: List[PrecomputedEmbeddings], top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Object search using pre-computed object_sparse vectors (no embed call)."""
        batches = []
        for emb in embeddings:
            if emb.object_sparse is None or not emb.object_sparse.indices:
                continue
            hits = self._query_points(
                query=emb.object_sparse, using=VEC_OBJECT_SPARSE,
                source="object", limit=top_k,
            )
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def search_ocr_with_vectors(
        self, embeddings: List[PrecomputedEmbeddings], top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """OCR search using pre-computed ocr_sparse vectors (no embed call)."""
        batches = []
        for emb in embeddings:
            if emb.ocr_sparse is None or not emb.ocr_sparse.indices:
                continue
            hits = self._query_points(
                query=emb.ocr_sparse, using=VEC_OCR_SPARSE,
                source="ocr", limit=top_k,
            )
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def search_metadata(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.metadata_enabled:
            return []

        title_field = self._discover_title_field()
        if title_field is None:
            if not self._warned_metadata:
                logger.warning(
                    "Metadata retrieval disabled — no title-like payload field in '%s'.",
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
                    hits.append({
                        "video_id": str(video_id),
                        "frame_id": int(frame_idx),
                        "score": 1.0 / rank,
                        "source": "metadata",
                        "raw_payload": payload,
                    })
                batches.append(hits)
            except Exception as exc:
                logger.warning("Metadata lexical retrieval failed for text=%r: %s", text, exc)

        return _merge_variant_results(batches, top_k=top_k)

    # ── internal ──────────────────────────────────────────────────────────
    def _search_dense_many(self, query_texts, using, encoder, source, top_k):
        batches = []
        for text in query_texts:
            vector = encoder(text)
            if vector is None:
                continue
            hits = self._query_points(query=vector, using=using, source=source, limit=top_k)
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def _search_sparse_many(self, query_texts, using, encoder, source, top_k):
        batches = []
        for text in query_texts:
            sparse_query = encoder(text)
            if sparse_query is None:
                continue
            hits = self._query_points(query=sparse_query, using=using, source=source, limit=top_k)
            batches.append(hits)
        return _merge_variant_results(batches, top_k=top_k)

    def _query_points(self, query, using, source, limit):
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
            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query,
                using=using,
                limit=limit,
                with_payload=self.payload_fields,
                with_vectors=False,
            )

        points = _extract_points(response)
        results = []
        for point in points:
            item = _payload_to_result(point, source=source)
            if item is not None:
                results.append(item)
        return results

    def _discover_title_field(self) -> Optional[str]:
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
            for f in POSSIBLE_TITLE_FIELDS:
                value = payload.get(f)
                if isinstance(value, str) and value.strip():
                    return f
        except Exception as exc:
            logger.warning("Failed to inspect metadata title field: %s", exc)
        return None

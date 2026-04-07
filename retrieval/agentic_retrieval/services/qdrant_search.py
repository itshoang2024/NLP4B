from __future__ import annotations

"""
qdrant_search.py — Qdrant-backed retrieval service for the Agentic Retrieval Branch.

Current collection design assumptions (aligned with qdrant_upsert.py):
- Single collection: keyframes_v1
- Named vectors in the same collection:
    * keyframe-dense           (SigLIP 1152d)
    * keyframe-caption-dense   (BGE-M3 1024d)
    * keyframe-object-sparse   (hashed sparse TF)
    * keyframe-ocr-sparse      (hashed sparse TF)
- Core payload fields per point:
    * video_id, frame_idx, azure_url, timestamp_sec, youtube_link
    * optional: tags, caption, detailed_caption, object_counts, ocr_text

Important note on metadata retrieval:
- The current upsert script does NOT create a metadata/title vector.
- If title is not stored in payload (e.g. title/video_title/youtube_title),
  search_metadata() will return [].
- For real semantic metadata retrieval, add either:
    (a) a dedicated title dense/sparse vector, or
    (b) an indexed title payload field + a representative-frame strategy.

Dependencies are expected to be installed from requirements.txt.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


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

SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
BGE_M3_MODEL = "BAAI/bge-m3"
BM25_MODEL = "Qdrant/bm25"

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


# ── model singletons ──────────────────────────────────────────────────────────
_bm25_model: Optional[SparseTextEmbedding] = None
_bge_m3_model: Optional[SentenceTransformer] = None
_siglip_tokenizer: Optional[Any] = None
_siglip_model: Optional[Any] = None
_siglip_device: Optional[str] = None


# ── text / vector helpers ─────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    values = [float(x) for x in vec]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return values
    return [v / norm for v in values]


def get_bm25() -> SparseTextEmbedding:
    """Lazy-load the same BM25 model used at upsert time (Qdrant/bm25)."""
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25: %s ...", BM25_MODEL)
        _bm25_model = SparseTextEmbedding(model_name=BM25_MODEL)
    return _bm25_model


def get_bge_m3() -> SentenceTransformer:
    global _bge_m3_model
    if _bge_m3_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading dense text model: %s on %s", BGE_M3_MODEL, device)
        _bge_m3_model = SentenceTransformer(BGE_M3_MODEL, device=device)
    return _bge_m3_model


def get_siglip_text_stack() -> Tuple[Any, Any, str]:
    global _siglip_tokenizer, _siglip_model, _siglip_device
    if _siglip_tokenizer is None or _siglip_model is None:
        _siglip_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading multimodal text model: %s on %s", SIGLIP_MODEL, _siglip_device)
        _siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL)
        _siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL).to(_siglip_device)
        _siglip_model.eval()
    return _siglip_tokenizer, _siglip_model, _siglip_device or "cpu"


# ── query encoders ────────────────────────────────────────────────────────────
def encode_sparse_query(text: str) -> Optional[models.SparseVector]:
    """Encode text using the same Qdrant/bm25 model used at upsert time."""
    text = _normalize_text(text)
    if not text:
        return None

    try:
        results = list(get_bm25().embed([text]))
        if not results:
            return None
        s = results[0]
        if len(s.indices) == 0:
            return None
        return models.SparseVector(
            indices=s.indices.tolist(),
            values=s.values.tolist(),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("BM25 sparse query encoding failed: %s", exc)
        return None


def encode_bge_m3_query(text: str) -> Optional[List[float]]:
    text = _normalize_text(text)
    if not text:
        return None

    try:
        vec = get_bge_m3().encode(text, normalize_embeddings=True)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]
    except Exception as exc:  # pragma: no cover
        logger.warning("BGE-M3 query encoding failed: %s", exc)
        return None


def encode_siglip_text_query(text: str) -> Optional[List[float]]:
    text = _normalize_text(text)
    if not text:
        return None

    try:
        tokenizer, model, device = get_siglip_text_stack()
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            raw = model.get_text_features(**inputs) if hasattr(model, "get_text_features") else model(**inputs)

            # Extract the 1152-d text embedding from whatever output format we get.
            if isinstance(raw, torch.Tensor):
                features = raw
            elif hasattr(raw, "pooler_output") and raw.pooler_output is not None:
                features = raw.pooler_output
            elif hasattr(raw, "text_embeds") and raw.text_embeds is not None:
                features = raw.text_embeds
            elif hasattr(raw, "last_hidden_state") and raw.last_hidden_state is not None:
                features = raw.last_hidden_state.mean(dim=1)
            else:
                features = raw[0]
                if getattr(features, "ndim", 0) == 3:
                    features = features.mean(dim=1)

        vec = features[0].detach().cpu().numpy().flatten().tolist()
        return _l2_normalize(vec)
    except Exception as exc:  # pragma: no cover
        logger.warning("SigLIP text query encoding failed: %s", exc)
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

    def __post_init__(self) -> None:
        self.url = self.url or os.environ.get("QDRANT_URL")
        self.api_key = self.api_key or os.environ.get("QDRANT_API_KEY")
        if not self.url or not self.api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY.")

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
            encoder=encode_siglip_text_query,
            source="keyframe",
            top_k=top_k,
        )

    def search_caption(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against BGE-M3 caption vectors using text queries."""
        return self._search_dense_many(
            query_texts=query_texts,
            using=VEC_CAPTION_DENSE,
            encoder=encode_bge_m3_query,
            source="caption",
            top_k=top_k,
        )

    def search_object(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against sparse object vectors."""
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OBJECT_SPARSE,
            encoder=encode_sparse_query,
            source="object",
            top_k=top_k,
        )

    def search_ocr(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search against sparse OCR vectors."""
        return self._search_sparse_many(
            query_texts=query_texts,
            using=VEC_OCR_SPARSE,
            encoder=encode_sparse_query,
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
